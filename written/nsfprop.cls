% nsfprop.cls
% William B. Thompson <thompson@cs.utah.edu>
% Current version: 2008-10-27
%
% A simple class file for typesetting NSF proposals.
%
% Default fonts and margins are for GPG09-1 (effective January, 2009):
% 11pt Computer Modern fonts and 1.0" margins.
%
%    Line spacing is ~5.25 lines / inch, well under the NSF
%    limit of 6 lines / inch for GPG09-1.  More text could be crammed
%    in without exceeding the NSF specification by modifying
%    \baselineskip, but would produce a hard-to-read document.
%
% Class options:
%
%    GPG07140
%
%      Use 2.5cm margins, as specified by GPG07-140 (effective June, 2007).
%
%    paragraphspacing
%
%      Switch from the default of indenting the start of each new
%      paragraph to an alternate format in which a blank line is placed
%      between paragraphs and there is no indenting at the start of each
%      paragraph. The default style allows more text to fit within the
%      15 page NSF limit.
%
%    timesfont
%
%      This option switches from 11pt Computer Modern to 11pt Times
%      Roman.  Times Roman is a denser font than ComputerModern in both
%      the characters themselves and the character spacing.  More text
%      will fit within the 15 page limit with Times Roman than with
%      ComputerModern.  Line spacing is as for Computer Modern.  You
%      should probably not use this option if the document uses
%      mathematics (see immediately below).
%
%    mathtimesfont
%
%      As with the timesfont option, except that mathematics is also set
%      in the times font.  See documentation on the mathptmx package for
%      information on when this might not work.
%
%    palatinofont
%
%      Typeset regular text and mathematics using 11 point Palantino.
%      See documentation on the mathpazo package for more information.
%      The Palantino font lays out similar to ComputerModern but with
%      substantially heavier glyphs.
%
%    palatinofont10
%
%      Typeset regular text and mathematics using 10 point Palantino.
%      See documentation on the mathpazo package for more information.
%      This produces substantially denser text (more text fitting into
%      the 15 page limit) than does the other options, but is allowed by
%      GPG 09-1
%
%    sfsectionfont
%
%	Switch to sans-serif font for section headings.
%
% Three standard LaTeX packages are loaded:
%
%    cite	% Sort numeric references in citations
%    version	% Mostly for the comment environment
%    xspace	% Used in macro definitions (but not in this class file!)
%
% Unlike previous versions of this class, no graphics bundle is loaded
% automatically.
%
% The figure placement rules used by LaTeX are turned off to give more
% control to the user.
%
% One new environments is defined:
%
%    \begin{localscope}
%
%		:
%
%    \end{localscope}
%
%	The localscope environment produces no formatting changes on its
%	own, but makes it easy to confine the scope of user-specified
%	formatting commands to a specific range.
%
%
% One new command is defined:
%
%    \boxtext{text}
%
%	Typeset the argument in a box prominently displayed on the page.
%	Useful for notes that need to be addressed before the document
%	is finalized.
%
\NeedsTeXFormat{LaTeX2e}[2001/06/01]
\ProvidesClass{nsfprop}[2008/10/27 v1.4 Document Class for NSF proposals]
%
\newif\if@tenpoint		% to support one option...
\newcommand{\loadtimesfont}{}	% Start with loading of times font unset.
\newcommand{\loadmathtimesfont}{}% Start with loading of mathtimes font unset.
\newcommand{\loadpalatinofont}{}% Start with loading of palatino font unset.
\newcommand{\resetsecfont}{}	% Start with sans-serif headings unset.
\newdimen\nsfmargin		% Allows quasi-metric or English margins.
\newcommand{\paragraphspacing}{}% Change default paragraph spacing
\setlength{\nsfmargin}{1.0in}	% Default margins (GPG09-1).
%
\DeclareOption{GPG081}{         % GPG081 uses English units.
	\setlength{\nsfmargin}{1.0in}
}
\DeclareOption{GPG07140}{	% GPG07140 uses quasi-metric margins
	\setlength{\nsfmargin}{2.5cm}
}
\DeclareOption{paragraphspacing}{ % Blank line between paragraphs, no indent
	\setlength{\parindent}{0em}\setlength{\parskip}{1.5ex}
	\renewcommand{\paragraphspacing}{%
		\setlength{\parindent}{0em}\setlength{\parskip}{1.5ex}%
	}
}
\DeclareOption{timesfont}{	% Switch to Times Roman from default CM
	\renewcommand{\loadtimesfont}{\usepackage{times}}
}
\DeclareOption{mathtimesfont}{	% Switch to Times Roman for both text and math.
	\renewcommand{\loadmathtimesfont}{\usepackage{mathptmx}}
}
\DeclareOption{palatinofont}{	% Switch to Palatino for both text and math.
	\renewcommand{\loadmathtimesfont}{\usepackage{mathpazo}}
}
\DeclareOption{palatinofont10}{	% Switch to Palatino for both text and math.
	\renewcommand{\loadmathtimesfont}{\usepackage{mathpazo}}
	\global\@tenpointtrue
}
\DeclareOption{sfsectionfont}{	% Switch to sans-serif font for section headings
	\renewcommand{\resetsecfont}{\usepackage[sf,bf,big]{titlesec}}
}
\ProcessOptions
%
\if@tenpoint
	\LoadClass[10pt]{article}	% Mostly, this is just the article class.
\else
	\LoadClass[11pt]{article}	% Mostly, this is just the article class.
\fi
%
\loadtimesfont			% If requested (there MUST be a better way!)
\loadmathtimesfont		% If requested (there MUST be a better way!)
\loadpalatinofont		% If requested (there MUST be a better way!)
\resetsecfont			% If requested (there MUST be a better way!)
\paragraphspacing		% If requested (there MUST be a better way!)
%
\usepackage{vmargin}		% Set up page layout
\setpapersize{USletter}
\setmarginsrb{\nsfmargin}{\nsfmargin}{\nsfmargin}{0.55\nsfmargin}{0pt}{0pt}%
{0pt}{0.45\nsfmargin}
%
\RequirePackage{cite}		% To order numerical citations (sortcite
% package no longer available!)
\RequirePackage{version}	% For the comment environment
\RequirePackage{xspace}		% We will need it for something...
%
% Allow for local changes of formatting variables
%
\newenvironment{localscope}{}{}
%
%
% Put text into a box centered on the page.
%
\newdimen\boxtextwidth
\newcommand{\boxtext}[1]{\hfill\rule{0ex}{0.01ex}
	\setlength{\boxtextwidth}{0.9\textwidth}
	\centerline{\fbox{\parbox{\boxtextwidth}{#1}}\vspace{1ex}}\newline}
%
% Turn off efforts by LaTeX to layout figures sensibly.  Figure placement
% will more sensitive to where the figure declaration occurs in the
% source.  This should be invoked before the \begin{document}
%
\setcounter{topnumber}{20}
\setcounter{bottomnumber}{20}
\setcounter{totalnumber}{20}
\renewcommand{\topfraction}{1.0}
\renewcommand{\bottomfraction}{1.0}
\renewcommand{\textfraction}{0.1}
%
