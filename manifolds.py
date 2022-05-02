import torch as th
import numpy as np
from torch.autograd import Function
from acosh import acosh


class Manifold:
    def __init__(self, manifold_type,  eps=1e-12, _eps=1e-5, norm_clip=1, max_norm=1e6, debug=False, K=None, **kwargs):
        self.manifold_type = manifold_type
        self.eps = eps
        self._eps = _eps
        self.norm_clip = norm_clip
        self.max_norm = max_norm
        self.debug = debug
        self.K = K
        if K is not None:
            self.inner_radius = 2 * self.K / (1 + np.sqrt(1 + 4 * self.K * self.K))

    def init_weights(self, w, scale=1e-4):
        w.weight.data.uniform_(-scale, scale)

    def distance(self, u, v):
        if self.manifold_type == 'euclidean':
            return ((u - v).pow(2)).sum(dim=-1)
        if self.manifold_type == 'lorentz':
            d = -LorentzDot.apply(u, v)
            d.data.clamp_(min=1)
            return acosh(d, self._eps)

    @staticmethod
    def ldot(u, v, keepdim=False):  # iloczyn skalarny w przestrzeni Lorentza
        uv = u * v
        uv.narrow(-1, 0, 1).mul_(-1)
        return th.sum(uv, dim=-1, keepdim=keepdim)

    def normalize(self, w):
        """Normalize vector such that it is located on the hyperboloid"""
        d = w.size(-1) - 1
        narrowed = w.narrow(-1, 1, d)
        if self.max_norm:
            narrowed.view(-1, d).renorm_(p=2, dim=0, maxnorm=self.max_norm)

        if self.K is not None:
            # Push embeddings outside of `inner_radius`
            w0 = w.narrow(-1, 0, 1).squeeze()
            wnrm = th.sqrt(th.sum(th.pow(narrowed, 2), dim=-1)) / (1 + w0)
            scal = th.ones_like(wnrm)
            ix = wnrm < (self.inner_radius + self._eps)
            scal[ix] = (self.inner_radius + self._eps) / wnrm[ix]
            narrowed.mul_(scal.unsqueeze(-1))

        tmp = 1 + th.sum(th.pow(narrowed, 2), dim=-1, keepdim=True)
        tmp.sqrt_()
        w.narrow(-1, 0, 1).copy_(tmp)
        return w

    def rgrad(self, p, d_p):
        """Riemannian gradient for hyperboloid"""
        if d_p.is_sparse:
            u = d_p._values()
            x = p.index_select(0, d_p._indices().squeeze())
        else:
            u = d_p
            x = p
        u.narrow(-1, 0, 1).mul_(-1)
        u.addcmul_(self.ldot(x, u, keepdim=True).expand_as(x), x)
        return d_p

    def expm(self, p, d_p, lr=None, out=None, normalize=False):
        """Exponential map for hyperboloid"""
        if out is None:
            out = p
        if d_p.is_sparse:
            ix, d_val = d_p._indices().squeeze(), d_p._values()
            # This pulls `ix` out of the original embedding table, which could
            # be in a corrupted state.  normalize it to fix it back to the
            # surface of the hyperboloid...
            # TODO: we should only do the normalize if we know that we are
            # training with multiple threads, otherwise this is a bit wasteful
            p_val = self.normalize(p.index_select(0, ix))
            ldv = self.ldot(d_val, d_val, keepdim=True)
            if self.debug:
                assert all(ldv > 0), "Tangent norm must be greater 0"
                assert all(ldv == ldv), "Tangent norm includes NaNs"
            nd_p = ldv.clamp_(min=0).sqrt_()
            t = th.clamp(nd_p, max=self.norm_clip)
            nd_p.clamp_(min=self.eps)
            newp = (th.cosh(t) * p_val).addcdiv_(th.sinh(t) * d_val, nd_p)
            if normalize:
                newp = self.normalize(newp)
            p.index_copy_(0, ix, newp)
        else:
            if lr is not None:
                d_p.narrow(-1, 0, 1).mul_(-1)
                d_p.addcmul_((self.ldot(p, d_p, keepdim=True)).expand_as(p), p)
                d_p.mul_(-lr)
            ldv = self.ldot(d_p, d_p, keepdim=True)
            if self.debug:
                assert all(ldv > 0), "Tangent norm must be greater 0"
                assert all(ldv == ldv), "Tangent norm includes NaNs"
            nd_p = ldv.clamp_(min=0).sqrt_()
            t = th.clamp(nd_p, max=self.norm_clip)
            nd_p.clamp_(min=self.eps)
            newp = (th.cosh(t) * p).addcdiv_(th.sinh(t) * d_p, nd_p)
            if normalize:
                newp = self.normalize(newp)
            p.copy_(newp)

    def logm(self, x, y):
        """Logarithmic map on the Lorenz Manifold"""
        xy = th.clamp(self.ldot(x, y).unsqueeze(-1), max=-1)
        v = acosh(-xy, self.eps).div_(
            th.clamp(th.sqrt(xy * xy - 1), min=self._eps)
        ) * th.addcmul(y, xy, x)
        return self.normalize_tan(x, v)


    def ptransp(self, x, y, v, ix=None, out=None):
        """Parallel transport for hyperboloid"""
        if ix is not None:
            v_ = v
            x_ = x.index_select(0, ix)
            y_ = y.index_select(0, ix)
        elif v.is_sparse:
            ix, v_ = v._indices().squeeze(), v._values()
            x_ = x.index_select(0, ix)
            y_ = y.index_select(0, ix)
        else:
            raise NotImplementedError
        xy = self.ldot(x_, y_, keepdim=True).expand_as(x_)
        vy = self.ldot(v_, y_, keepdim=True).expand_as(x_)
        vnew = v_ + vy / (1 - xy) * (x_ + y_)
        if out is None:
            return vnew
        else:
            out.index_copy_(0, ix, vnew)

class LorentzDot(Function):
    @staticmethod
    def forward(ctx, u, v):
        ctx.save_for_backward(u, v)
        return Manifold.ldot(u, v)

    @staticmethod
    def backward(ctx, g):
        u, v = ctx.saved_tensors
        g = g.unsqueeze(-1).expand_as(u).clone()
        g.narrow(-1, 0, 1).mul_(-1)
        return g * v, g * u


if __name__ == '__main__':
    m = Manifold(None)
    print(m)
    t1 = th.tensor([1, 2, 3])
    t2 = th.tensor([4, 5, 6])
    print(m.ldot(t1, t2))
