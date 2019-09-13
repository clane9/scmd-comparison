function [params] = set_default_params(params, fields, defaults)
for ii=1:length(fields)
  if ~isfield(params, fields{ii})
    params.(fields{ii}) = defaults{ii};
  end
end
end
