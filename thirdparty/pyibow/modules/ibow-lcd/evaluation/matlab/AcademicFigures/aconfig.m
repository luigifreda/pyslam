% ACONFIG Get an Academic Figure configuration.
%    config = ACONFIG returns a default configuration for an 
%             Academic Figure.
%
%    config = ACONFIG(PropertyName, PropertyValue, ...) returns the 
%             configuration and sets values for some properties.
%
% The properties of the configuration are:
%
%    BackgroundColor (default: [1 1 1]): figure background color.
%    FontSize (default: 20): title, axes, labels and text font size.
%    LineWidth (default: 2): width of plot lines.
%    DrawBox (defaut: true): show a full box around around the axes.
%    Grid (default: true): show grid.
%    RemoveMargins (default: true): remove blank space around axes.
%    Width (default: []): figure width in pixels. Ignored if empty.
%    Height (default: []): figure height in pixels. Ignored if empty.
%    SizeRatio (default: 4/3): width by height ratio of window. Ignored if
%       empty of it both Width and Height are given.
%    Colormap (default: 'cmr'): colormap. It can be a Nx3 matrix with 
%       values in [0, 1] representing N RGB colors, or the name of a 
%       predefined map (these maps keep high contrast when converted into
%       grayscale): 'thermal', 'cmr', 'dusk', 'hsv2', 'gray'.
%    LineStyles (default: '-|-.|--'): styles of plot lines.
%
%    See also afigure, aplot, abar
function config = aconfig(varargin)
  config.BackgroundColor = [1 1 1];
  config.FontSize = 20;
  config.LineWidth = 2;
  config.DrawBox = true;
  config.Grid = true;
  config.RemoveMargins = true;
  config.Width = [];
  config.Height = [];
  config.SizeRatio = 4/3; 
  config.Colormap = 'cmr';
  config.LineStyles = '-|-.|--';  

  for i = 1:2:numel(varargin)
    item = varargin{i};
    if isfield(config, item)
      if i + 1 > numel(varargin),
        warning('AcademicFigures:aconfig', ...
          'No value given for property %s', item);
      else
        config.(item) = varargin{i+1};
      end
    else
      warning('AcademicFigures:aconfig', 'Unknown field: %s', item);
    end
  end
  