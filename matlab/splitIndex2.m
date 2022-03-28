function [indexCell,idx] = splitIndex2(idxx)

    idx = [1 find(diff(idxx') ~= 1)+1  numel(idxx')+1];
    indexCell = mat2cell(idxx', 1, diff(idx))';     
    
    end