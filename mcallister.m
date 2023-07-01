function [ucblist, lcblist]=mcallister(x,p_posterior,p_prior, cases, delta)
KL_term = get_KL(p_posterior,p_prior, cases);
[repetition, T]=size(x);

conf_term = log((T+2)/delta);

is_monte_carlo = 0;
%emp_term = sum(p_posterior*mean(x,2));
switch cases
    case 'gaussian'
        emp_term=mean(x,'all');
        is_monte_carlo=1;
    otherwise
        emp_term = dot(p_posterior,mean(x,2));
end

monte_carlo_term=sqrt(log(1/delta)/2/repetition);
width = sqrt((KL_term + conf_term)/2/(T-1))+is_monte_carlo*monte_carlo_term;
ucblist=min(emp_term+width,1);
lcblist=max(emp_term-width,0);
end