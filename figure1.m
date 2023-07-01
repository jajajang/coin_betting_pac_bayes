p_prior=[0.2,0.8];
p_posterior=[0.1,0.9];
p_sample=0.5;
theta=[0,1];
delta=0.01;

repetitions=10;
n_rounds=16;
history_baseline=zeros(1,n_rounds);
history_lower=zeros(repetitions,n_rounds);
history_upper=zeros(repetitions,n_rounds);
history_mcalister_lower=zeros(repetitions,n_rounds);
history_mcalister_upper=zeros(repetitions,n_rounds);
history_london_lower=zeros(repetitions,n_rounds);
history_london_upper=zeros(repetitions,n_rounds);
history_alquier_lower=zeros(repetitions,n_rounds);
history_alquier_upper=zeros(repetitions,n_rounds);
history_emp_bern_lower=zeros(repetitions,n_rounds);
history_emp_bern_upper=zeros(repetitions,n_rounds);
history_klver_lower=zeros(repetitions,n_rounds);
history_klver_upper=zeros(repetitions,n_rounds);

lossy = @(x_, theta_) transpose(theta_)*x_;


n_total=2^n_rounds;



for z=1:repetitions
tic
X_total=binornd(1,p_sample,1,n_total);
display(X_total(1:2))
for c=1:n_rounds
n_samples=2^(c-1);

X=X_total(1:n_samples);

mu_0=mean(lossy(X,theta),2);
%optimization for our algorithm
B = @(p_s,p_0,n) get_KL(p_s,p_0,'discrete')+gammaln(1/2)+gammaln(n+1)-gammaln(n+1/2)-log(delta);
f = @(mu_) dot(p_posterior,mu_); %since we use fmin we calculate minus of the true objective
f_neg = @(mu_) -dot(p_posterior,mu_); %since we use fmin we calculate minus of the true objective
g = @(mu_) sum(psi_star(mu_,X,theta,lossy).*p_posterior) - B(p_posterior,p_prior,n_samples);
gfun = @(mu_) deal(g(mu_),[]);
h = @(mu_) dot(get_KL_bunch(mu_0,mu_, n_samples),p_posterior) - B(p_posterior,p_prior,n_samples);
hfun = @(mu_) deal(h(mu_),[]);

cons = @(mu_) max(g(mu_), h(mu_));
consfun = @(mu_) deal(cons(mu_),[]);

Vn_emp=dot(p_posterior, var(lossy(X,theta),0,2));
%history_baseline(c)=dot(p_posterior,mu_0);
history_baseline(c)=p_sample*(dot(theta,p_posterior));

options = optimoptions('fmincon','Algorithm','interior-point','Display','iter');
[mu_upper, funcval_upper]=fmincon(f_neg,ones(length(theta),1)/2,[],[],[],[],zeros(length(theta),1),ones(length(theta),1),gfun,options);
[mu_lower, funcval_lower]=fmincon(f,ones(length(theta),1)/2,[],[],[],[],zeros(length(theta),1),ones(length(theta),1),gfun,options);
[nu_upper, funcval_upper_nu]=fmincon(f_neg,ones(length(theta),1)/2,[],[],[],[],zeros(length(theta),1),ones(length(theta),1),hfun,options);
[nu_lower, funcval_lower_nu]=fmincon(f,ones(length(theta),1)/2,[],[],[],[],zeros(length(theta),1),ones(length(theta),1),hfun,options);
history_upper(z,c)=-funcval_upper;
history_lower(z,c)=funcval_lower;
history_klver_upper(z,c)=-funcval_upper_nu;
history_klver_lower(z,c)=funcval_lower_nu;
if c>1
    history_upper(z,c)=min(history_upper(z,c-1), -funcval_upper);
    history_lower(z,c)=max(history_lower(z,c-1),funcval_lower);
    history_klver_upper(z,c)=min(history_klver_upper(z,c-1), -funcval_upper_nu);
    history_klver_lower(z,c)=max(history_klver_lower(z,c-1), funcval_lower_nu);
end


[history_mcalister_upper(z,c), history_mcalister_lower(z,c)]=mcallister(lossy(X,theta),p_posterior,p_prior,'discrete',delta);
[history_london_upper(z,c), history_london_lower(z,c)]=benlondon(lossy(X,theta),p_posterior,p_prior,'discrete',delta);
[history_alquier_upper(z,c), history_alquier_lower(z,c)]=alquier(lossy(X,theta),p_posterior,p_prior,'discrete',delta);
[history_emp_bern_upper(z,c), history_emp_bern_lower(z,c)]=empirical_bernstein(lossy(X,theta),p_posterior,p_prior,Vn_emp,'discrete',delta);
end

toc
end
%plotting
save('230210_bernoulli_final')

x = linspace(1,n_rounds,n_rounds)-1;
linewidth=1;
p7=plot(x,mean(history_klver_upper),color='#06C2AC',LineWidth=linewidth);

hold on
p2=plot(x,mean(history_mcalister_upper),'blue',LineWidth=linewidth);
p2l=plot(x,mean(history_mcalister_lower),'blue',LineWidth=linewidth);
p3=plot(x,mean(history_london_upper),color="#77AC30",LineWidth=linewidth);
p3l=plot(x,mean(history_london_lower),color="#77AC30",LineWidth=linewidth);
p4=plot(x,mean(history_alquier_upper),color="#7E2F8E",LineWidth=linewidth);
p4l=plot(x,mean(history_alquier_lower),color="#7E2F8E",LineWidth=linewidth);
p5=plot(x,history_baseline,'black',LineWidth=linewidth, LineStyle='--');
p6=plot(x,mean(history_emp_bern_upper),color='#D95319',LineWidth=linewidth);
p6l=plot(x,mean(history_emp_bern_lower),color='#D95319',LineWidth=linewidth);
p7l=plot(x,mean(history_klver_lower),color='#06C2AC',LineWidth=linewidth);

p1=plot(x,mean(history_upper), 'red',LineWidth=linewidth);
p1l=plot(x,mean(history_lower),'red',LineWidth=linewidth);
hold off
legend([p1,p2,p3,p4,p6,p5,p7],'Ours', 'Mcallister', 'London', 'Maurer','Emp bern', 'True value', 'Kl-ver')
xlabel('log_2(n)')
ylabel('Bound')



% Now its time to do other toy case. theta_raw~N(0,1) and discretize it. 
% f(X,theta)=erf (X*theta)
% logistic regression in 2d

% Ben london
% Mcalister
% Alquier intro 3.3
%========================== after it works well
% PAC-Bayes Un-Expected Bernstein Inequality
% PAC-Bayes-empirical-Bernstein inequality