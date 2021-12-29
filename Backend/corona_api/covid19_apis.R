library(httr)
library(jsonlite)
library(dotenv)
library(psych)

load_dot_env()

url <- "https://vaccovid-coronavirus-vaccine-and-treatment-tracker.p.rapidapi.com/api/covid-ovid-data/sixmonth/USA"
API_KEY <- Sys.getenv("RAPID_API_KEY")
HOST_URL <- 'vaccovid-coronavirus-vaccine-and-treatment-tracker.p.rapidapi.com'
response <- VERB("GET", url,
                 httr::add_headers("x-rapidapi-host" = HOST_URL,
                                   "x-rapidapi-key" = API_KEY), 
                 content_type("application/octet-stream"))

df <- fromJSON(rawToChar(response$content))

head(df)
describe(df)
