// ============================================================
// Custom math macros - ported from LaTeX config
// ============================================================

// --- Lowercase bold vectors ---
#let vx = $bold(x)$
#let vy = $bold(y)$
#let vz = $bold(z)$
#let vf = $bold(f)$
#let vw = $bold(w)$
#let vu = $bold(u)$
#let vv = $bold(v)$
#let va = $bold(a)$
#let vb = $bold(b)$
#let vc = $bold(c)$
#let vj = $bold(j)$
#let vm = $bold(m)$
#let vn = $bold(n)$
#let vp = $bold(p)$
#let vq = $bold(q)$
#let vr = $bold(r)$
#let vs = $bold(s)$
#let vt = $bold(t)$
#let vk = $bold(k)$
#let vh = $bold(h)$
#let ve = $bold(e)$
#let vJ = $bold(J)$
#let vG = $bold(G)$

// --- Uppercase bold matrices ---
#let vA = $bold(A)$
#let vB = $bold(B)$
#let vC = $bold(C)$
#let vD = $bold(D)$
#let vF = $bold(F)$
#let vH = $bold(H)$
#let vI = $bold(I)$
#let vK = $bold(K)$
#let vL = $bold(L)$
#let vM = $bold(M)$
#let vP = $bold(P)$
#let vQ = $bold(Q)$
#let vR = $bold(R)$
#let vS = $bold(S)$
#let vU = $bold(U)$
#let vV = $bold(V)$
#let vW = $bold(W)$
#let vX = $bold(X)$
#let vY = $bold(Y)$
#let vZ = $bold(Z)$

// --- Greek bold vectors ---
#let valpha   = $bold(alpha)$
#let vbeta    = $bold(beta)$
#let vgamma   = $bold(gamma)$
#let vdelta   = $bold(delta)$
#let vepsilon = $bold(epsilon)$
#let veta     = $bold(eta)$
#let vlambda  = $bold(lambda)$
#let vmu      = $bold(mu)$
#let vnu      = $bold(nu)$
#let vphi     = $bold(phi)$
#let vpsi     = $bold(psi)$
#let vsigma   = $bold(sigma)$
#let vtheta   = $bold(theta)$
#let vxi      = $bold(xi)$
#let vomega   = $bold(omega)$
#let vtau     = $bold(tau)$
#let vell     = $bold(ell)$

// --- bold zero vector ---
#let vzero = $bold(0)$

// --- Greek bold matrices ---
#let vSigma  = $bold(Sigma)$
#let vLambda = $bold(Lambda)$
#let vGamma  = $bold(Gamma)$
#let vPhi    = $bold(Phi)$
#let vPsi    = $bold(Psi)$
#let vOmega  = $bold(Omega)$
#let vTheta  = $bold(Theta)$

// --- Mathematical operators ---
#let E    = $bb(E)$
#let R    = $bb(R)$
#let NN   = $cal(N)$    // \mathcal{N}  (N is a built-in in Typst math)
#let GP   = $cal(G) cal(P)$
#let Var  = $op("Var")$
#let Cov  = $op("Cov")$
#let KL   = $op("KL")$
#let Tr   = $op("tr")$
#let Diag = $op("diag")$

// --- Calculus shorthands ---
// Usage: $dd(f, x)$  →  df/dx
#let dd(num, den)  = $frac(d #num, d #den)$
#let pd(num, den)  = $frac(partial #num, partial #den)$
#let pdd(num, den) = $frac(partial^2 #num, partial #den^2)$

// --- Misc ---
#let defeq = $:=$
#let iid   = $attach(tilde, t: "iid")$
#let ind   = $attach(tilde, t: "ind")$

// ============================================================
// Section numbering: level 1 → "1.0", level 2 → "1.1",
//                   level 3 → "1.1.1", deeper → "1.1.1.1" …
// ============================================================
#set heading(numbering: (..nums) => {
  let n = nums.pos()
  if n.len() == 1 {
    str(n.at(0)) + ".0"
  } else {
    n.map(str).join(".")
  }
})
