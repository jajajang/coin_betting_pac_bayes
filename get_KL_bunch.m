function kl = get_KL_bunch(mu_0, mu_1, n_samples)
n=length(mu_0);    
kl=zeros(1,n);
    for i=1:n
        kl(i)= n_samples*get_KL([1-mu_0(i), mu_0(i)], [1-mu_1(i), mu_1(i)],'bernoulli');
    end
end
