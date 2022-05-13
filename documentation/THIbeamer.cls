\ProvidesClass{THIbeamer}

% Handle the aspectratio input--------------------------------------
\newcommand{\THIaspectratio}{43}
\DeclareOption{169}{\renewcommand{\THIaspectratio}{169}}
\DeclareOption{43}{\renewcommand{\THIaspectratio}{43}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{beamer}}
\ProcessOptions\relax

%Load base document class and required packages--------------------
\LoadClass[t,aspectratio=\THIaspectratio]{beamer}
\RequirePackage{tikz}
\RequirePackage{xcolor}
\RequirePackage[scaled]{helvet}
\RequirePackage[T1]{fontenc}
\RequirePackage{setspace}
\setstretch{1,3}

\definecolor{THIblue}{RGB}{0,90,155}
\definecolor{AININgreen1}{RGB}{207,244,42}
\definecolor{AININgreen}{RGB}{154,183,36}

\setbeamersize{text margin left=5mm,text margin right=5mm} 

% Deactivate navigation symbols --> conflict with page #-----------
\setbeamertemplate{navigation symbols}{}

% Set colors and symbols for titles, enumeration and items---------
\setbeamercolor{frametitle}{fg=AININgreen}
\setbeamercolor{title}{fg=AININgreen}
\setbeamercolor{footline}{fg=AININgreen}
\setbeamercolor{caption name}{fg=AININgreen}  
\setbeamercolor{enumerate item}{fg=AININgreen}  
\setbeamercolor{itemize item}{fg=AININgreen}
\setbeamercolor{enumerate subitem}{fg=AININgreen}  
\setbeamercolor{itemize subitem}{fg=AININgreen!80}
\setbeamercolor{itemize subsubitem}{fg=AININgreen!60}  
\setbeamertemplate{itemize item}[circle]
\setbeamertemplate{itemize subitem}[circle]
\setbeamertemplate{itemize subsubitem}[circle]

% Set colors and symbols bib----------------------------------------
\setbeamertemplate{bibliography item}{\insertbiblabel}
\setbeamertemplate{bibliography entry title}{}
\setbeamertemplate{bibliography entry location}{}
\setbeamertemplate{bibliography entry note}{}
\setbeamercolor{bibliography item}{fg=AININgreen}
\setbeamercolor*{bibliography entry title}{fg=gray}
\setbeamercolor*{bibliography entry location}{fg=gray}
\setbeamercolor*{bibliography entry author}{fg=black}

% Set the footline---------------------------------------------------
\setbeamertemplate{footline}[text line]{%
\hspace{0.02\paperwidth}
  \parbox[b][12pt][c]{0.7\paperwidth}{
    \inserttitle ~--~ \insertauthor
  }
  \hfill%
 \parbox[b][12pt][c]{0.15\paperwidth}{
    \raggedleft\insertframenumber ~/~ \inserttotalframenumber
  }
}

% Set the background for common frames-----------------------------
\setbeamertemplate{background}{%
\ifnum\THIaspectratio=43
\noindent\parbox{0.8\paperwidth}{\vspace*{1.7cm}\begin{tikzpicture}
\draw[fill= AININgreen,color=AININgreen](0,0){[rounded corners=2pt]--
	++(0.3,0)}{[rounded corners=8pt]--
	++(0,7.5)}--
	++(-0.3,0)--
	cycle;
\end{tikzpicture}}
\fi
\ifnum\THIaspectratio=169
\noindent\parbox{0.8\paperwidth}{\vspace*{1.7cm}\begin{tikzpicture}
\draw[fill= AININgreen,color=AININgreen](0,0){[rounded corners=2pt]--
	++(0.3,0)}{[rounded corners=8pt]--
	++(0,6.9)}--
	++(-0.3,0)--
	cycle;
\end{tikzpicture}}
\fi
}

% Set the title for common frames----------------------------------
\setbeamertemplate{frametitle}
{
    \nointerlineskip
        \vbox{}
        \vskip0.25ex%
        \strut{\color{AININgreen}{{\insertframetitle}}}\strut
        \hfill
		\parbox[r]{6ex}{\includegraphics[scale=0.1]{required/ainin.png}}
		\strut{\color{AININgreen}\\[-1.3ex]{{{\large \insertframesubtitle}}}}\strut
        \vskip-2ex%
}


% Set the front frame----------------------------------------------
\pgfdeclareimage[height=\paperheight]{mybackground}{required/plainback.pdf}
\pgfdeclareimage[width=0.15\paperwidth]{KIVIlogo}{required/kivi.png}
\pgfdeclareimage[width=0.15\paperwidth]{AININlogo}{required/ainin.png}


\setbeamertemplate{title page}{
\ifnum\THIaspectratio=43
	\begin{picture}(0,0)
            \put(-60,-268){%
                \pgfuseimage{mybackground}
            }
            \put(110,-268){%
            	\tikz\draw[fill=white,color=white,opacity=0.85,line width=0pt](3,0)--(9,0)--(9,-10)--(0,-10) -- cycle;
            }
            \put(160,-120){%
                \begin{minipage}[t]{62mm}
                \raggedleft
                    {\color{AININgreen}\Large \inserttitle}\\
                    {\color{black!70}\small \insertsubtitle}\\[2ex]
                    {\color{AININgreen}\small \insertauthor \quad \insertdate}
                \end{minipage}
            }
            \put(270,-255){%
                \pgfuseimage{KIVIlogo}
            }
            \put(270,-40){%
                \pgfuseimage{AININlogo}
           	}
	\end{picture}
\fi
\ifnum\THIaspectratio=169
	\begin{picture}(0,0)
            \put(-14,-250){%
                \pgfuseimage{mybackground}
            }
            \put(170,-268){%
            	\tikz\draw[fill=white,color=white,opacity=0.85,line width=0pt](3,0)--(10,0)--(10,-10)--(0,-10) -- cycle;
            }
            \put(225,-110){%
                \begin{minipage}[t]{71mm}
                \raggedleft
                    {\color{AININgreen}\Large \inserttitle}\\
                    {\color{black!70}\small \insertsubtitle}\\[2ex]
                    {\color{AININgreen}\small \insertauthor \quad \insertdate}
                \end{minipage}
            }
            \put(355,-240){%
                \pgfuseimage{KIVIlogo}
            }
            \put(355,-50){%
                \pgfuseimage{AININlogo}
           	}
	\end{picture}
\fi
}
    
