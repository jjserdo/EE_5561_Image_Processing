function x = cg_solve(b, A, no_iter)

if nargin < 3
    no_iter = 3;
end

x0 = zeros(size(b), 'single');
x = x0;
r=b-A(x);
p=r;
rsold=r(:)'*r(:);

for ind = 1:no_iter
    Ap = A(p);
    alpha = rsold/(p(:)'*Ap(:));
    x = x + alpha*p;
    r = r - alpha*Ap;
    rsnew = r(:)'*r(:);
    p = r + rsnew/rsold*p;
    rsold = rsnew;
end