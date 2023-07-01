function kl = get_KL(p_1, p_2, cases)
kl=0;
switch cases
    case 'bernoulli'
        if p_1(1)==0
            if p_2(2)==0
                kl=Infty;
            else
                kl=log(1/p_2(2));
            end
        elseif p_1(1)==1
            if p_2(1)==0
                kl=Infty;
            else
                kl=log(1/p_2(1));
            end
        else
            kl=sum(p_1.*(log(p_1./p_2)));
        end
    case 'gaussian'
        kl=0.5*log(p_2(2)/p_1(2)) + ((p_1(1)-p_2(1))^2+p_1(2))/(2*p_2(2)) - 0.5;
    case 'discrete'
        kl=sum(p_1.*(log(p_1./p_2)));
    otherwise
        print('Wrong cases')
end
end
