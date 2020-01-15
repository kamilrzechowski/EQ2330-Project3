function quantized = quantizer(x,step)
    %%%%%%%%%%%%
    % formula for Mid-thred quantizer from:
    % https://dsp-nbsphinx.readthedocs.io/en/nbsphinx-experiment/quantization/linear_uniform_characteristic.html
    %%%%%%%%%%%%
    
    quantized = step*round(x/step);
end

