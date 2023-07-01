function [ucblist, lcblist]=empirical_bernstein(x,p_posterior,p_prior, precise_v_n, cases, delta)
KL_term = get_KL(p_posterior,p_prior, cases);
c1=1.1;
c2=1.1;
[repetition, T]=size(x);
vari = var(x,0,2);

conf_term = log(1/delta);
nu = ceil(1/log(c1)*sqrt((exp(1)-2)*T/4/conf_term))+1;

conf_term2=log(nu)+conf_term + log(2);

is_monte_carlo=0;
switch cases
    case 'gaussian'
        emp_term=mean(x,'all');
        is_monte_carlo=1;
    otherwise
        emp_term = dot(p_posterior,mean(x,2));
end

nu_2 = 1/log(c2)*log(0.5*sqrt((T-1)/log(1/delta)+1)+0.5);
common_part=(KL_term + log(nu_2/delta))/(T-1);
exp_V_n = precise_v_n+(1+c2)*sqrt(precise_v_n*common_part/2)+2*c2*common_part;

bar_V_n = min(exp_V_n, 1/4);

%emp_term = sum(p_posterior*mean(x,2));

width=(1+c1)*sqrt((exp(1)-2)*bar_V_n*(KL_term+conf_term2)/T) + is_monte_carlo*sqrt(log(1/delta)/repetition);

ucblist=min(emp_term+width,1);
lcblist=max(emp_term-width,0);
end