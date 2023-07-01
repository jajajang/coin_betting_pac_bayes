function [ucblist, lcblist]=alquier_montecarlo(x,p_posterior,p_prior, cases, delta)
[repetition, T]=size(x);


% Maurer's bound with monte-carlo sampling
% This code will be mainly used for Figure 3 and 4. 

KL_term = get_KL(p_posterior,p_prior, cases);
conf_term = log(4*sqrt(T)/delta);


% indicator for monte-carlo operation
is_monte_carlo=0;

switch cases
    case 'gaussian'
        emp_term=mean(x,'all');
        is_monte_carlo=1;
    otherwise
        emp_term = dot(p_posterior,mean(x,2));
end

% Function for the monte-carlo bound constraint (Proposition 14)

bound_monte=log(4/delta)/repetition;
h_up = @(mu_) get_KL([1-emp_term, emp_term],[1-mu_,mu_],'bernoulli')-bound_monte;
h_low = @(mu_) get_KL([1-emp_term, emp_term],[1-mu_,mu_],'bernoulli')-bound_monte;

% binary search for monte-carlo operation - Line 5 of Algorithm 2

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


% Function for the PAC-Bayes bound constraint

g_up = @(mu_) get_KL([1-emp_term_upper, emp_term_upper],[1-mu_,mu_],'bernoulli')-(KL_term+conf_term)/T;
g_low = @(mu_) get_KL([1-emp_term_lower, emp_term_lower],[1-mu_,mu_],'bernoulli')-(KL_term+conf_term)/T;

% binary search for PAC-Bayes bound - Line 6 of Algorithm 2

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