library(casnet)
library(ggplot2)
library(plyr)
library(dplyr)

x <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

rm <- rp(x)

rm

meltRP <- reshape2::melt(rm)

gRP <-  ggplot2::ggplot(ggplot2::aes_(x=~Var1, y=~Var2, fill = ~value), data= meltRP) +
  ggplot2::geom_raster(hjust = 0, vjust=0, show.legend = TRUE) +
  ggplot2::geom_abline(slope = 1,colour = "grey50", size = 1)

gRP + ggplot2::scale_fill_gradient2(low = "red3",
                              high     = "steelblue",
                              mid      = "white",
                              na.value = scales::muted("slategray4"),
                              midpoint = mean(meltRP$value, na.rm = TRUE),
                              limit    = c(min(meltRP$value, na.rm = TRUE),max(meltRP$value, na.rm = TRUE)),
                              space    = "Lab",
                              name     = "")


