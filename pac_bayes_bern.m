function [ucblist, lcblist]=pac_bayes_bern(x,p_posterior,p_prior, precise_v_n, cases, delta)

n_details=100;
inc=1/n_details;
f = @(c1) pac_bayes_bern_c1(x,p_posterior,p_prior, precise_v_n, cases, delta, c1);
c1=1.1;
f_mini=f(c1);
f_chal=100;
stop_sign=true;

while stop_sign
    c1=c1+inc;
    f_chal=f(c1);
    if f_chal<f_mini
        f_mini=f_chal;
    else
        stop_sign=false;
    end
end


emp_term=0;
is_monte_carlo=0;
switch cases
    case 'gaussian'
        emp_term=mean(x,'all');
        is_monte_carlo=1;
    otherwise
        emp_term = dot(p_posterior,mean(x,2));
end

ucblist=min(emp_term + f_mini,1);
lcblist=max(emp_term - f_mini,0);

end