function [ucblist, lcblist]=alquier(x,p_posterior,p_prior, cases, delta)
[repetition, T]=size(x);


KL_term = get_KL(p_posterior,p_prior, cases);
conf_term = log(2*sqrt(T)/delta);

is_monte_carlo=0;

%emp_term = sum(p_posterior*mean(x,2));
switch cases
    case 'gaussian'
        emp_term=mean(x,'all');
        is_monte_carlo=1;
    otherwise
        emp_term = dot(p_posterior,mean(x,2));
end

f= @(mu_) get_KL([1-emp_term, emp_term],[1-mu_,mu_],'bernoulli');
g = @(mu_) get_KL([1-emp_term, emp_term],[1-mu_,mu_],'bernoulli')-(KL_term+conf_term)/T - is_monte_carlo*sqrt(log(1/delta)/repetition);

%binary search

mu_max=1-eps;
mu_min=eps;

if g(mu_max)<0
    mu_max_challenge=mu_max;
else
    mu_max_challenge=emp_term;
end

if g(mu_min)<0
    mu_min_challenge=mu_min;
else
    mu_min_challenge=emp_term;
end

while abs(mu_max-mu_max_challenge)>10^(-7)
    mu_cache=(mu_max+mu_max_challenge)/2;
    if g(mu_cache)>0
        mu_max=mu_cache;
    else
        mu_max_challenge=mu_cache;
    end
end

while abs(mu_min-mu_min_challenge)>10^(-7)
    mu_cache=(mu_min+mu_min_challenge)/2;
    if g(mu_cache)>0
        mu_min=mu_cache;
    else
        mu_min_challenge=mu_cache;
    end
end



ucblist=mu_max;
lcblist=mu_min;
end