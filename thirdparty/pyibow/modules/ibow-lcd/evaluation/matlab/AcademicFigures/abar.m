% ABAR Bar for Academic Figures.
%    ABAR(...) does the same as bar, but creates an Academic Figure if
%    there is not any active figure.
%
%    handles = ABAR(...) returns the handles of the plotted bars.
%
%    See also afigure, aconfig, aplot
function handles = abar(varargin)
  afigure(gcf);
  hnds = bar(varargin{:});
  if nargout > 0
    handles = hnds;
  end
