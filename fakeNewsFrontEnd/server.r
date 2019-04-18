# Define server logic to read selected file ----
server <- function(input, output,session) {
  
  #install.packages("readtext")
  library("readtext")
  create_readData_file<-"D:/RWebProject/ShinyApp-ver1/ver03/readData.txt"
  algorithm_file_path<-"D:/RWebProject/ShinyApp-ver1/ver03/action.R"
  create_Result_file<- "D:/RWebProject/ShinyApp-ver1/ver03/result.txt"
  
    #on submit button condition
    observeEvent(input$submit, {

      # if input text and no file upload then show messgae dialog
      if(input$text =="" && is.null(input$file1))
       {
        showModal
         (modalDialog(
          title = "Results",
          "Either File is not uploaded or text field is empty",
          easyClose = TRUE
          ))
        #showmodel closed
       }
      #if closed 
      
      #separate  condition
      # check if text input is filled
      
      else if(input$text!=""){
        #x<-"D:/RWebProject/ShinyApp-ver1/ver03/readData.txt"
        
        #write.table(input$text, file=x,row.names=FALSE,col.names = FALSE,sep="")
        
        #save input-text into readData.txt
        sink(create_readData_file)
        cat(input$text)
        sink()
        
        #output section'
        output$contents <- renderTable({
        
        #run R Script this section is to call algorithm scripts like xgboost,sentimental analysis  
        
        source(algorithm_file_path)
          #provide here R Script of algorithms here
          
        #after script is run it saves result into result.txt
        result_File<- create_Result_file
          
        resultant<-readtext(result_File)
        
        #this section is for shwing data from result.txt 
        print(resultant$text)
        }) 
        #output closed
        # show dialog about final result like fake or not
        showModal(modalDialog(
          title = "Results",
          "The New is FAKE!",
          easyClose = TRUE
        )) 
        #sho wmodal closed
      } 
      #else if closed
      #check if file is uploaded 
      # provided with input-text is empty 
      # and submit button is clicked
      else if(!(is.null(input$file1))){
        output$contents <- renderTable({

           req(input$file1)
          tryCatch(
            {
              #copy file with same name (readData.txt)
             # current.folder<-"D:/RWebProject/ShinyApp-ver1/ver03/"
              #file.copy(input$file1$datapath,paste0(current.folder,"readData.txt"), overwrite = TRUE)
              
              file.copy(input$file1$datapath,create_readData_file, overwrite = TRUE)
              
              #run R Script this section is to call algorithm scripts like xgboost,sentimental analysis
              source(algorithm_file_path)

              #after script is run it saves result into result.txt
              result_File<- create_Result_file

              resultant<-readtext(result_File)
              #this section is for shwing data from result.txt
              print(resultant$text)

            },
            # try closed
            error = function(e) {
              # return a safeError if a parsing error occurs
              stop(safeError(e))
            }
            #error closed
          )
          #trycatch closed
        })
        #output$contents- renderTable closed
      }
      # #else if - fileInput closed
      # 
    
      

  }) 
  #server closed
  
}