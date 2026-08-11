function v = envopt(name, dflt)
% ENVOPT  Numeric value from environment variable NAME, or DFLT if unset.
% Lets discretisation be overridden for convergence studies without editing
% the model.
s = getenv(name);
if isempty(s), v = dflt; else, v = str2double(s); end
end
