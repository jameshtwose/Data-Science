\name{apple_mobility_data}
\alias{apple_mobility_data}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
%%  ~~function to do ... ~~
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
A function to get the most up to date version of the apple mobility data (documentation can be found here: https://covid19.apple.com/mobility).
}
\usage{
apple_mobility_data()
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{None}{
}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
A data frame object.
}
\references{
%% ~put references to the literature/web site here ~
Apple maps: Mobility Trends Reports

https://covid19.apple.com/mobility
}
\author{
%%  ~~who you are~~
James Twose (https://services.jms.rocks)
}
\note{
This function is a translation of the following python script written by Ilari Patrikka:


\code{import json}

\code{from urllib.request import urlopen}

\code{def main():}

   \code{url = "https://covid19-static.cdn-apple.com/covid19-mobility-data/current/v3/index.json"}

   \code{response = urlopen(url)}

   \code{data = json.loads(response.read())}

   \code{url = ("https://covid19-static.cdn-apple.com" + data['basePath'] + data['regions']['en-us']['csvPath'])}

   \code{return url}
   
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
df <- apple_mobility_data()

}

