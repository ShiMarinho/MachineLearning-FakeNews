# Define UI for data upload app ----
ui <- fluidPage(
  
  # App title ----
  titlePanel(" Fake News Detector"),
  
  # Sidebar layout with input and output definitions ----
  sidebarLayout(
    
    # Sidebar panel for inputs ----
    sidebarPanel(
 
      #text input     
      textInput(inputId = "text", label = "Text input"
                
      ),
      
      #submit button with name submit
      actionButton("submit","Submit"),
#      actionButton("reset_input","Reset"),
      
      
      # Horizontal line ----
      tags$hr(),
      
      
      # Input: Select a file ----
      fileInput("file1", "File input",
                multiple = FALSE,
                accept = c("text",
                           "text/plain"
                           )),
      

      
      #showing Note 
      p("Note: Use Text field or Upload file:"),
      
      # Horizontal line ----
      tags$hr()
      
      
    ),   # slidebar panel closed
    
    # Main panel for displaying outputs ----
    mainPanel(
      
      # Output: Data file ----
      tableOutput("contents"),
      
      #warning for wrong file upload
      h1(textOutput("warning1"))
      
    ) # main panel closed
    
  ) # slidebar layout closed
) # fluid page closed