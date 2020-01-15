function OUT = my_dct2(block)
% computes dct of the block

if(size(block,1) ~= 8 && size(block,2)~= 8)
    error('block size should be 8x8');
    OUT = []
    return
end
    
M = size(block,1);

for k=0:M-1
    for i=0:M-1
        alpha = sqrt(((i~=0)+1)/M);
        c = cos(((2*k+1)*i*pi)/(2*M));
        A(i+1,k+1) = alpha * c; %stupid matlab non-zero indexing
    end
end

A = dctmtx(M);

OUT = A*X*A';
end

