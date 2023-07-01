function [ucblist, lcblist]=benlondon(x,p_posterior,p_prior, cases, delta)



KL_term = get_KL(p_posterior,p_prior, cases);
[repetition, T]=size(x);
conf_term = log(T/delta);

is_monte_carlo = 0;
switch cases
    case 'gaussian'
        emp_term=mean(x,'all');
        is_monte_carlo=1;
    otherwise
        emp_term = dot(p_posterior,mean(x,2));
end


width1=2*(KL_term+conf_term)/(T-1);

func= @(tau) sqrt(width1*(emp_term-1+(1/tau))/tau) + width1/tau;

monte_carlo_term=sqrt(log(1/delta)/repetition);
width=func(1)+is_monte_carlo*monte_carlo_term;
ucblist=min(emp_term+width,1);
lcblist=max(emp_term-width,0);
end