% APLOT Plot for Academic Figures.
%    APLOT(...) does the same as plot, but creates an Academic Figure if
%    there is not any active figure.
%
%    handles = APLOT(...) returns the handles of the plotted lines.
%
%    See also afigure, aconfig, abar
function handles = aplot(varargin)
  afigure(gcf);
  hnds = plot(varargin{:});
  if nargout > 0
    handles = hnds;
  end
