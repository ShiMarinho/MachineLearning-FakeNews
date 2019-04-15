#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

#install all packages used in all R Scripts in the begining
install.packages("shiny")
#install.packages("readtext")
#use libraries for all scripts here
#library("readtext")
library("shiny")



source("ui.R")
source("server.R")
# Run the application 
shinyApp(ui = ui, server = server)

