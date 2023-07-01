function [ucblist, lcblist]=alquier_montecarlo(x,p_posterior,p_prior, cases, delta)
[repetition, T]=size(x);

disp(repetition)
disp(T)
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

%First monte-carlo bound

%emp_term_upper=emp_term+is_monte_carlo*sqrt(log(1/delta)/2/repetition);
%emp_term_lower=emp_term-is_monte_carlo*sqrt(log(1/delta)/2/repetition);
bound_monte=log(2/delta)/repetition;
h_up = @(mu_) get_KL([1-emp_term, emp_term],[1-mu_,mu_],'bernoulli')-bound_monte;
h_low = @(mu_) get_KL([1-emp_term, emp_term],[1-mu_,mu_],'bernoulli')-bound_monte;
%binary search

mu_max_monte=1-eps;
mu_min_monte=eps;
mu_max_challenge_monte=emp_term;
mu_min_challenge_monte=emp_term;

while abs(mu_max_monte-mu_max_challenge_monte)>10^(-7)
    mu_cache=(mu_max_monte+mu_max_challenge_monte)/2;
    if h_up(mu_cache)>0
        mu_max_monte=mu_cache;
    else
        mu_max_challenge_monte=mu_cache;
    end
end

while abs(mu_min_monte-mu_min_challenge_monte)>10^(-7)
    mu_cache=(mu_min_monte+mu_min_challenge_monte)/2;
    if h_low(mu_cache)>0
        mu_min_monte=mu_cache;
    else
        mu_min_challenge_monte=mu_cache;
    end
end

emp_term_upper=mu_max_monte;
emp_term_lower=mu_min_monte;


%Second PAC bound
g_up = @(mu_) get_KL([1-emp_term_upper, emp_term_upper],[1-mu_,mu_],'bernoulli')-(KL_term+conf_term)/T;
g_low = @(mu_) get_KL([1-emp_term_lower, emp_term_lower],[1-mu_,mu_],'bernoulli')-(KL_term+conf_term)/T;
%binary search

mu_max=1-eps;
mu_min=eps;
mu_max_challenge=emp_term_upper;
mu_min_challenge=emp_term_lower;

while abs(mu_max-mu_max_challenge)>10^(-7)
    mu_cache=(mu_max+mu_max_challenge)/2;
    if g_up(mu_cache)>0
        mu_max=mu_cache;
    else
        mu_max_challenge=mu_cache;
    end
end

while abs(mu_min-mu_min_challenge)>10^(-7)
    mu_cache=(mu_min+mu_min_challenge)/2;
    if g_low(mu_cache)>0
        mu_min=mu_cache;
    else
        mu_min_challenge=mu_cache;
    end
end



ucblist=mu_max;
lcblist=mu_min;
end