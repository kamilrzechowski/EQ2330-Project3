function [recoBlock, resRate] = residualCoding(movBlock,orgBlock, q_step)
    
    b_size = size(movBlock,1);
    residual = orgBlock - movBlock;
    residualDCT = comp_dct(residual,8);
    blockDCTq = quantizer(residualDCT,q_step);
    resBlock = comp_idct(blockDCTq,8);
    recoBlock = resBlock + movBlock;
    
    %vectors of DCT coefficients
    vals = reshape(blockDCTq(:,:),[1,size(blockDCTq(:,:),1)*size(blockDCTq(:,:),2)]);
    % compute bins to estimate pdfs
    bins_coefs = min(vals):q_step:max(vals);
    % histogram with bins to get pdfs
    pdf8x8 = hist(vals,bins_coefs)/length(vals);
    % compute entropy from pdfs
    H = -sum(pdf8x8.*log2(pdf8x8+eps));
    resRate = H*(b_size^2);
end