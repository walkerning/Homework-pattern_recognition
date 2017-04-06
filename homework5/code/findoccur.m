function [ occurs ] = findoccur( array, nums )
%FINDOCCUR
    occurs = arrayfun(@(num) length(find(array==num)), nums);
end

