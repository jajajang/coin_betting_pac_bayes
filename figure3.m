p_prior=[0,1];
p_posterior=[0,0.25];
p_dist=[0,1];
n_group=4;
delta=0.05;
rng(1);
multiplier=1/delta^(1/n_group);


tic

n_rounds=12;
theta_all=normrnd(p_posterior(1), p_posterior(2), 1, 2^n_rounds);
n_samples=2^5;
history_baseline=zeros(1,n_rounds);
history_lower=ones(1,n_rounds);
history_upper=zeros(1,n_rounds);
history_mcalister_lower=zeros(1,n_rounds);
history_mcalister_upper=zeros(1,n_rounds);
history_london_lower=zeros(1,n_rounds);
history_london_upper=zeros(1,n_rounds);
history_alquier_lower=zeros(1,n_rounds);
history_alquier_upper=zeros(1,n_rounds);
history_alquier_lower_h=zeros(1,n_rounds);
history_alquier_upper_h=zeros(1,n_rounds);
history_emp_bern_lower=zeros(1,n_rounds);
history_emp_bern_upper=zeros(1,n_rounds);

lossy = @(x_, theta_) (erf(transpose(theta_)*x_)+1)*0.5;


n_total=2^n_rounds;
X_total=normrnd(p_dist(1), p_dist(2),1,n_total);
Vn_emp=precise_expect_vn(p_dist, p_posterior,lossy,10000, 10000,'gaussian');
for c=4:n_rounds-2
    N_theta=2^c;
    display(N_theta)
    n_batch_size=N_theta/n_group;
    X=X_total(1:n_samples);
    mu_0=mean(lossy(X,theta_all(1:N_theta)),2);
    %optimization for our algorithm
    epsi=0.01;
    
    theta=theta_all(1:N_theta);

    new_upper_pre=0;
    new_lower_pre=1;
    
    for batch=1:n_group
        theta=theta_all((batch-1)*n_batch_size+1:batch*n_batch_size);
        B = @(p_s,p_0,n) get_KL(p_s,p_0,'gaussian')+gammaln(1/2)+gammaln(n+1)-gammaln(n+1/2)-log(delta/3);
        
        %f = @(mu_) dot(p_posterior,mu_); %since we use fmin we calculate minus of the true objective
        %f_neg = @(mu_) -dot(p_posterior,mu_); %since we use fmin we calculate minus of the true objective
        
        f = @(mu_) mean(mu_); %since we use fmin we calculate minus of the true objective
        f_neg = @(mu_) -mean(mu_); %since we use fmin we calculate minus of the true objective
        
        g = @(mu_) (mean(psi_star(mu_,X,theta,lossy)) - B(p_posterior,p_prior,n_samples))*multiplier;
        gfun = @(mu_) deal(g(mu_),[]);
        
        mu_0_batch=mu_0((batch-1)*n_batch_size+1:batch*n_batch_size);
        
        options = optimoptions('fmincon','Algorithm','interior-point','Display','iter', 'MaxFunctionEvaluations',600000);
        [mu_upper, funcval_upper]=fmincon(f_neg,(ones(length(theta),1)+mu_0_batch)/2,[],[],[],[],zeros(length(theta),1),ones(length(theta),1),gfun,options);
        [mu_lower, funcval_lower]=fmincon(f,mu_0_batch/2,[],[],[],[],zeros(length(theta),1),ones(length(theta),1),gfun,options);
        
        if new_upper_pre<-funcval_upper
            new_upper_pre=-funcval_upper;
        end
        if new_lower_pre>funcval_lower
            new_lower_pre=funcval_lower;
        end
    end
    
    [history_upper(c), history_lower(c)]=bin_search(p_prior, p_posterior, 'gaussian', new_upper_pre,new_lower_pre,n_batch_size,delta/3);
    
    %history_upper(c)=new_upper_pre;
    %history_lower(c)=new_lower_pre;


    history_baseline(c)=mean(mu_0);
    
    %[history_mcalister_upper(c), history_mcalister_lower(c)]=mcallister(lossy(X,theta_all),p_posterior,p_prior,'gaussian',delta);
    %[history_london_upper(c), history_london_lower(c)]=benlondon(lossy(X,theta_all),p_posterior,p_prior,'gaussian',delta);
    [history_alquier_upper(c), history_alquier_lower(c)]=alquier_montecarlo(lossy(X,theta),p_posterior,p_prior,'gaussian',delta);
    [history_alquier_upper_h(c), history_alquier_lower_h(c)]=alquier_montecarlo_hoeffding(lossy(X,theta),p_posterior,p_prior,'gaussian',delta);
    %[history_emp_bern_upper(c),history_emp_bern_lower(c)]=empirical_bernstein(lossy(X,theta_all),p_posterior,p_prior,Vn_emp,'gaussian',delta);
    %[history_emp_bern_upper(c),history_emp_bern_lower(c)]=pac_bayes_bern(lossy(X,theta_all),p_posterior,p_prior,Vn_emp,'gaussian',delta);
end

toc

save('230630_changing_m')
%plotting
x = linspace(1,n_rounds,n_rounds)-2;
linewidth=1;
guideline=ones(1,n_rounds)*sqrt(log(1/delta)/2/N_theta);

start=4;

p1=plot(x(start:n_rounds-2),history_upper(start:n_rounds-2), 'red',LineWidth=linewidth);
%titlestring=sprintf('Monte Carlo Gaussian Plot, postvar=%.2f, E[V(\theta)]=%f',p_posterior(2),Vn_emp);
%title(titlestring)

%history_emp_bern_upper(1)=1;
%history_emp_bern_lower(1)=0;

hold on
p1l=plot(x(start:n_rounds-2),history_lower(start:n_rounds-2),'red',LineWidth=linewidth);
%p2=plot(x,history_mcalister_upper,'blue',LineWidth=linewidth);
%p2l=plot(x,history_mcalister_lower,'blue',LineWidth=linewidth);
%p3=plot(x,history_london_upper,color="#77AC30",LineWidth=linewidth);
%p3l=plot(x,history_london_lower,color="#77AC30",LineWidth=linewidth);
p4=plot(x(start:n_rounds-2),history_alquier_upper(start:n_rounds-2),color="#7E2F8E",LineWidth=linewidth);
p4l=plot(x(start:n_rounds-2),history_alquier_lower(start:n_rounds-2),color="#7E2F8E",LineWidth=linewidth);
%p4h=plot(x(start:n_rounds-2),history_alquier_upper_h(start:n_rounds-2),color="#77AC30",LineWidth=linewidth);
%p4hl=plot(x(start:n_rounds-2),history_alquier_lower_h(start:n_rounds-2),color="#77AC30",LineWidth=linewidth);
p5=plot(x(start:n_rounds-2),history_baseline(start:n_rounds-2),'black',LineWidth=linewidth, LineStyle='--');
%p6=plot(x,history_emp_bern_upper,color=[0.75,0.75,0],LineWidth=linewidth);
%p6l=plot(x,history_emp_bern_lower,color=[0.75,0.75,0],LineWidth=linewidth);
%p7=plot(x,history_baseline+guideline,color='black', LineStyle='--');
%p7l=plot(x,history_baseline-guideline,color='black', LineStyle='--');
hold off
legend([p1,p4, p5],'Ours', 'Maurer','True mean')
xlabel('log_2(m)')
ylabel('Bound')

%history_baseline=ones(1,n_rounds)*mean(mu_0);



% Now its time to do other toy case. theta_raw~N(0,1) and discretize it. 
% f(X,theta)=erf (X*theta)
% logistic regression in 2d

% Ben london
% Mcalister
% Alquier intro 3.3
%========================== after it works well
% PAC-Bayes Un-Expected Bernstein Inequality
% PAC-Bayes-empirical-Bernstein inequality