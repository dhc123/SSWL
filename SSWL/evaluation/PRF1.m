function [ Precision,Recall,F1 ] = PRF1(X,Y,varargin)
%SLPRF Calculate the average micro Precision, recall and F1 measure
% $ Syntax $
%   - slprf( X,Y,varargin)

%
% $ Description $
%   - X: the classifier's output labels for sample i : Outputs(:,i)with predicted label=1 else -1
%   - Y: the real labels, each column for one sample , +1 for label, else -1 
%   - 'type':   the type of P,R,F: 
%               'micro'--micro-averaging (results are computed based on global sums over all decisions) (default ='micro')
%               'macro'--macro-averaging (results are computed on a
%               per-category basis, then averaged over categories)
%       Micro-averaged scores tend to be dominated by the most commonly used categories, 
%       while macro-averaged scores tend to be dominated by the performance in rarely used categories. 
%   
% $ History $
%   - Created by Xiangnan Kong, on Jan 7, 2008
%
%% parse and verify input arguments
opts.type='micro';
opts = slparseprops(opts, varargin{:});
%% calculate average Precision
X(X>0) = 1;X(X<=0) = 0;
Y(Y>0) = 1;Y(Y<=0) = 0;
XandY = X&Y;
if strcmp(opts.type,'micro')
    Precision=sum(XandY(:))/sum(X(:));
    if isnan(Precision)
        Precision=0;
    end
    Recall=sum(XandY(:))/sum(Y(:));
    F1=2*Precision*Recall/(Precision+Recall);
end
if strcmp(opts.type,'marco')
    p=sum(XandY,1)./sum(X,1);
    r=sum(XandY,1)./sum(Y,1);
    

    tmp_p=isnan(p);
    
    if sum(double(tmp_p))>0
        AAA=1;
    end
    p(tmp_p)=0;
    f=2*p.*r./(p+r);
    
    tmp_f=isnan(f);
    f(tmp_f)=0;
    
    Precision = mean(p);
    Recall = mean(r);
    F1 = mean(f);
    if isnan(F1)
        AAAA=1;
    end
    
end
