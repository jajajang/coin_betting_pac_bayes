function [betstar, fval] = find_max_log_wealth_constrained(g,bmin,bmax)

myf = @(bet) sum(log(1 + g.*bet));
df = @(bet) sum(g./(1 + g.*bet));
df2 = @(bet) -sum((g./(1 + g.*bet)).^2);

%[betstar, fval] = fminbnd_faster2(myf, df, max(bmin,-1e10), min(bmax,1e10));
[betstar, fval] = newton_1d_bnd(myf, df, df2, max(bmin,-1e10), min(bmax,1e10));
%fval = -fval;
