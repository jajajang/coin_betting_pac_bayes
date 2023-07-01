function result = psi_star(mu, X, theta, f)

result=zeros(1,length(theta));
loss=f(X,theta);


for i=1:length(theta)

bmax=1/(mu(i)+eps);
bmin=-1/(1-eps-mu(i));

g=loss(i,:)-mu(i);

myf = @(bet) sum(log(1 + g.*bet));
df = @(bet) sum(g./(1 + g.*bet));
df2 = @(bet) -sum((g./(1 + g.*bet)).^2);
[betstar, fval] = newton_1d_bnd(myf, df, df2, max(bmin,-1e10), min(bmax,1e10));

result(i)=fval;

end