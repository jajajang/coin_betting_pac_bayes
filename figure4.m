p_prior=[0,1];
p_posterior=[0,0.25];
p_dist=[0,1];
n_group=4;
delta=0.05;
rng(1);
multiplier=(3/delta)^(1/n_group);


n_rounds=5;
N_theta=2^10;
theta_all=normrnd(p_posterior(1), p_posterior(2), 1, N_theta);
history_baseline=zeros(1,n_rounds);
history_lower=ones(1,n_rounds);
history_upper=zeros(1,n_rounds);
history_alquier_lower=zeros(1,n_rounds);
history_alquier_upper=zeros(1,n_rounds);

lossy = @(x_, theta_) (erf(transpose(theta_)*x_)+1)*0.5;


n_total=2^n_rounds;
X_total=normrnd(p_dist(1), p_dist(2),1,n_total);
Vn_emp=precise_expect_vn(p_dist, p_posterior,lossy,10000, 10000,'gaussian');

for c=1:n_rounds
    n_samples=2^(c-1);
    n_batch_size=N_theta/n_group;
    X=X_total(1:n_samples);
    mu_0=mean(lossy(X,theta_all(1:N_theta)),2);
    %optimization for our algorithm
    epsi=0.01;
    
    new_upper_pre=0;
    new_lower_pre=1;
    
    for batch=1:n_group
        tic
        theta=theta_all((batch-1)*n_batch_size+1:batch*n_batch_size);
        B = @(p_s,p_0,n) get_KL(p_s,p_0,'gaussian')+gammaln(1/2)+gammaln(n+1)-gammaln(n+1/2)-log(delta/3);
                
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

        toc
    end
    
    [history_upper(c), history_lower(c)]=bin_search(p_prior, p_posterior, 'gaussian', new_upper_pre,new_lower_pre,n_batch_size,delta/3);
    

    history_baseline(c)=mean(mu_0);
    
    [history_alquier_upper(c), history_alquier_lower(c)]=alquier_montecarlo(lossy(X,theta_all),p_posterior,p_prior,'gaussian',delta);
    
end
%plotting

save('230629_workspace_changing_n_large_m')

x = linspace(1,n_rounds,n_rounds)-1;
linewidth=1;
guideline=ones(1,n_rounds)*sqrt(log(1/delta)/2/N_theta);

p1=plot(x,history_upper, 'red',LineWidth=linewidth);

hold on
p1l=plot(x,history_lower,'red',LineWidth=linewidth);
p4=plot(x,history_alquier_upper,color="#7E2F8E",LineWidth=linewidth);
p4l=plot(x,history_alquier_lower,color="#7E2F8E",LineWidth=linewidth);
p5=plot(x,history_baseline,'black',LineWidth=linewidth, LineStyle='--');
hold off
legend([p1,p4, p5],'Ours', 'Maurer','True mean')
xlabel('log_2(n)')
ylabel('Bound')