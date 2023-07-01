function [x,fval] = newton_1d_bnd(funfcn,df,df2,ax,bx)

deriv=inf;
x=0;
x_old=inf;
while abs(deriv)>0.001 && abs(x-x_old)>0.001
    x_old=x;
    deriv=df(x);
    if x==ax && deriv<0
        break
    elseif x==bx && deriv>0
        break
    end
    if abs(deriv)>1e3
        update=0.01*sign(deriv);
    else
        deriv2=df2(x);
        update=-deriv/deriv2;
    end
    x=x+update;
    x=max(min(x,bx),ax);
end
fval=funfcn(x);

