function [reco_block, rate] = codeBlock(block_in, q_step, b_size)
    blockDCT = comp_dct(block_in,8);
    blockDCTq = quantizer(blockDCT,q_step);
    reco_block = comp_idct(blockDCTq,8);
    
    %vectors of DCT coefficients
    vals = reshape(blockDCTq(:,:),[1,size(blockDCTq(:,:),1)*size(blockDCTq(:,:),2)]);
    % compute bins to estimate pdfs
    bins_coefs = min(vals):q_step:max(vals);
    % histogram with bins to get pdfs
    pdf8x8 = hist(vals,bins_coefs)/length(vals);
    % compute entropy from pdfs
    H = -sum(pdf8x8.*log2(pdf8x8+eps));
    rate = H*(b_size^2);
end