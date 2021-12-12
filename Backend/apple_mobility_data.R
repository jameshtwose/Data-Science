library(httr)

# A function to get the most up to date version of the apple mobility data (documentation can be found here: https://covid19.apple.com/mobility)

# This function is a translation of the following python script written by Ilari Patrikka:
# import json
# from urllib.request import urlopen
# def main():
#   url = "https://covid19-static.cdn-apple.com/covid19-mobility-data/current/v3/index.json"
#   response = urlopen(url)
#   data = json.loads(response.read())
#   url = ("https://covid19-static.cdn-apple.com" + data['basePath'] + data['regions']['en-us']['csvPath'])
#   return url


apple_mobility_data <- function(){
  initial_url <- "https://covid19-static.cdn-apple.com/covid19-mobility-data/current/v3/index.json"
  begin_path <- "https://covid19-static.cdn-apple.com"
  response <- GET(url = initial_url)
  data <- content(response)
  base_path <- data$basePath
  extension_path <- data$regions$`en-us`$csvPath
  url <- paste0(begin_path, base_path, extension_path)
  df <- read.csv(url)
  return(df)  
}

