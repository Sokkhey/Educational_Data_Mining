library(shiny)
library(shinydashboard)
library(shinyWidgets)
library(ggplot2)
library(DT)
library(dplyr)
library(class) # for knn
library(ModelMetrics)
library(randomForest)
library(caret)
library(corrplot)
#--------------------------------------
library(Boruta)
library(DEoptimR)
library(GPArotation)
#---------------------------------------

load("my_model")
D <- read.csv("EducationalData.csv")[,c("SIM1","SELD3","ANXI3","SELD5","SIM2","ANXI1","TMP3","SIM3","ANXI2","SIM4","SCORE")]
train_data <- read.csv("EducationalData.csv")[,c("SIM1","SELD3","ANXI3","SELD5","SIM2","ANXI1","TMP3","SIM3","ANXI2","SIM4","SCORE")]
grade_classifier <- function(x){
  prediction <- rep(NA, length(x)) # initialize prediction vector
  prediction[x >= 110] <- 3
  prediction[x >= 90 & x < 110] <- 2
  prediction[x >= 75 & x < 90] <- 1
  prediction[x < 75] <- 0
  return(factor(prediction, levels = c("0", "1", "2","3"))) # prediction is either 0 or 1
}
train_data$SCORE <- grade_classifier(train_data$SCORE)
##################################################################################################################
#NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN--------- Create UI -----------------------
##################################################################################################################

#---------------------------------------- header
header <- dashboardHeader(title = "System Menu",
                          titleWidth = 200,
                          dropdownMenu(
                            type = "notifications",
                            notificationItem(
                              text = "Educational Institutions",
                              icon = icon("school")
                            )
                          )
) #-- end header 
#-----------------------------------------side bar

sidebar <- dashboardSidebar(disable = FALSE, width = 200,
                            #----------------------------
                            sidebarMenu(
                              menuItem('Introduction',tabName = "introduction",icon = icon("dashboard")),
                              menuItem('Data Description',tabName = "samples",icon = icon("chart-bar")),
                              menuItem("Prediction (Student)",tabName = "prediction", icon = icon("user-graduate")),
                              menuItem("Prediction (School)",tabName = "prediction1", icon = icon("school"))
                            )
)
#---------------------------------------------------------------------------------------------------- body
body <- dashboardBody(
  tags$head(
    tags$style(
      HTML('
            h3 {
    font-weight: bold;
}
            ')
    )
  ),
  #-------------------------------------------------------------------
  tabItems(
    #--------------------------------------------------TabItem 1
    tabItem(tabName = "introduction",
            
            fluidRow(
              box(width = 12,title = "Introduction", status = 'info', #solidHeader = TRUE,
                  div("This system is used to predict the performance level of students in mathematics, so that the result can be use for intervention and improve thier performance.",
                      style = "color:navy"))
            ),
          
            fluidRow(infoBox(width = 6,title = 'Characteristics of Training Dataset:',color = 'green',
                             icon = icon("database"),"Database1"),
                     infoBox(width = 6,title = 'Characteristics of Training Dataset:',color = 'green',
                             icon = icon("database"),"Database1"),
                     infoBox(width = 6,title = 'Characteristics of Training Dataset:',color = 'green',
                            icon = icon("database"),"Database1"),
                     infoBox(width = 6,title = 'Characteristics of Training Dataset:',color = 'green',
                     icon = icon("database"),"Database1")),
            
            #--------------------------------
            fluidRow(
                     valueBox(value = 0,width = 3,
                              subtitle = "<60% :Poor student need high intervention",
                              icon = icon("database")),
                     valueBox(value = 1, width = 3,
                              subtitle = "60 - 75%: Average student need intervention",
                              icon = icon("database")),
                     valueBox(value = 2, width = 3,
                              subtitle = "75 - 90%: Good student need low intervention",
                              icon = icon("database")),
                     valueBox(value = 3, width = 3,
                              subtitle = ">= 90%: Excellent student no need intervention",
                              icon = icon("database"))
                     
            )
    ),
    #---------------------------------------------- TabItem 2
    tabItem(tabName = "samples",
            fluidRow(infoBox(width = 12,title = 'Characteristics of Training Dataset:',color = 'blue',
                             icon = icon("database"),
                             br(),
                             selectInput("features","All input features here", choices = c("SIM1"=1,"SELD3"=2,"ANXI3"=3,"SELD5"=4,"SIM2"=5,
                                                                                           "ANXI1"=6,"TMP3"=7,"SIM3"=8,"ANXI2"=9,"SIM4"=10,"SCORE"=11))),
                     tabsetPanel(type = "tab",
                                 tabPanel("Data",DT::DTOutput("data")),
                                 tabPanel("Summary",verbatimTextOutput("summary")),
                                 tabPanel("Structure",verbatimTextOutput("str")),
                                 tabPanel('Plot',plotOutput('myplot'))
                     ) 
            )
    ),
    #--------------------------------------------TabItem 3
    tabItem(tabName = "prediction",
            #-----------------------------------------------------------
            fluidRow(box(width = 12,title = "Please Enter or Answer the Question Below:","Instruction: Our input features use 5-point likert scale. For example for level of interest (1=Not interest at all, 2=Slightly interest, 3=Fairly interest,
                                                                                                                                                                                         4=Very interest, 5=Extremly interest) and for level of frequency (1= never, 2= rarely, 3= sometime, 4=often, 5= always),
                         and level of agreement (1=Strongly disagree, 2=Disagree, 3=Neutral, 4=Agree, 5= Strongly Agree)", status = 'primary',icon = icon("address-card"))
            ),
            #NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN Input Feature
            fluidRow(box(width = 4, title = "Name", status = 'info', solidHeader = TRUE,
                         textInput('name','Please enter your name',"")),
              box(width = 4,title = "ID number", status = 'info', solidHeader = TRUE,
                            textInput('ID','Please enter your ID',"")),
              #--------------------------------------------------------------------
              box(width = 4,title = "Q1: How much do you interest in Math?", status = 'info', solidHeader = TRUE,
                            sliderInput("SIM1",label = div(style = 'width:490px', div(style = 'float:left','Never'),
                                                div(style = 'float:right','Always')), min = 1, max = 5, value = 3, width = '500px')),
              box(width = 4,title = "Q2: How often do you do your math homework?", status = 'info', solidHeader = TRUE,
                           sliderInput("SELD3",label = div(style = 'width:490px', div(style = 'float:left','Never'),
                                                 div(style = 'float:right','Always')), min = 1, max = 5, value = 2, width = '500px')),
              box(width = 4,title = "Q3: How often do you feel helpless in math class?", status = 'info', solidHeader = TRUE,
                            sliderInput("ANXI3",label = div(style = 'width:490px', div(style = 'float:left','Never'),
                                                  div(style = 'float:right','Always')), min = 1, max = 5, value = 5, width = '500px')),
              box(width = 4,title = "Q4: How often do you prepare for math exam?", status = 'info', solidHeader = TRUE,
                  sliderInput("SELD5",label = div(style = 'width:490px', div(style = 'float:left','Never'),
                                                  div(style = 'float:right','Always')), min = 1, max = 5, value = 2, width = '500px')),
              box(width = 4,title = "Q5: How often do you feel enjoyable in  taking math class?", status = 'info', solidHeader = TRUE,
                  sliderInput("SIM2",label = div(style = 'width:490px', div(style = 'float:left','Never'),
                                                  div(style = 'float:right','Always')), min = 1, max = 5, value = 3, width = '500px')),
              box(width = 4,title = "Q6: How often do you feel tense and bored in math class?", status = 'info', solidHeader = TRUE,
                  sliderInput("ANXI1",label = div(style = 'width:490px', div(style = 'float:left','Never'),
                                                 div(style = 'float:right','Always')), min = 1, max = 5, value = 3, width = '500px')),
              box(width = 4,title = "Q7: you think your math teacher try to use several method to explain you?", status = 'info', solidHeader = TRUE,
                  sliderInput("TMP3",label = div(style = 'width:490px', div(style = 'float:left','Never'),
                                                  div(style = 'float:right','Always')), min = 1, max = 5, value = 4, width = '500px')),
              box(width = 4,title = "Q8: Do you think you always pay attention in math class?", status = 'info', solidHeader = TRUE,
                  sliderInput("SIM3",label = div(style = 'width:490px', div(style = 'float:left','Never'),
                                                 div(style = 'float:right','Always')), min = 1, max = 5, value = 3, width = '500px')),
              box(width = 4,title = "Q9: How often do you feel panic in math exam?", status = 'info', solidHeader = TRUE,
                  sliderInput("ANXI2",label = div(style = 'width:490px', div(style = 'float:left','Never'),
                                                 div(style = 'float:right','Always')), min = 1, max = 5, value = 4, width = '500px')),
              box(width = 4,title = "Q10: How often do you motivate yourself in studying math?", status = 'info', solidHeader = TRUE,
                  sliderInput("SIM4",label = div(style = 'width:490px', div(style = 'float:left','Never'),
                                                 div(style = 'float:right','Always')), min = 1, max = 5, value = 3, width = '500px')),
              
              box(width = 4,title = "Score in Matheamtics", status = 'info', solidHeader = TRUE,
                  br(),
                  textInput('score','Please enter your previous math score',""),
                  br())
            ),
          #  submitButton('Submit'),
          #  br(),
            #-------------------------------------------------------
            fluidRow(box(width = 12,title = "Overview of Your Input",status = 'primary',icon = icon("edit"),
                         br(),
                                    DT::DTOutput("yourdata"))
            ),
            #----------------------------------- Predict button
            actionButton("go","Predict Your Performance Level", icon("paper-plane"), 
                         style="color: #fff; background-color: #00a65a; border-color: #00a65a"),
                       # verbatimTextOutput("predvalue"),
            br(),
            br(),
            #----------------------- Accuracy Result
            fluidRow(box(width = 12,title = "Performance Level Result:", status = 'primary', #color = "green",icon = icon("user-graduate"),
                             verbatimTextOutput("predvalue"),
                             tableOutput("pl")))
            #------------------------------------------------------
    ),
    #--------------------------------------------------------------------------Tab Item 4
    tabItem(tabName = 'prediction1',
            fluidRow(
              box(title="File Upload", fileInput("file1", "Choose CSV File"), status = "success", solidHeader = TRUE, width = "12"),
              box(title = 'Select Input Features',status = 'primary',width = 12,
                  selectInput("features4","All input features here", choices = c("SIM1"=1,"SELD3"=2,"ANXI3"=3,"SELD5"=4,"SIM2"=5,
                                                                                "ANXI1"=6,"TMP3"=7,"SIM3"=8,"ANXI2"=9,"SIM4"=10,"SCORE"=11)),
              tabsetPanel(type = "tab",
                          #tabPanel("Data",DT::dataTableOutput("contents4")),
                          tabPanel('Plot',plotOutput('myplot4')),
                          tabPanel("Summary",verbatimTextOutput("summary4")),
                          tabPanel("Corrplot",plotOutput("corrplot4")))
              )
            ),
            br(),
            br(),
            #-------------------------------------
            fluidRow(box(width = 12,title = "Actual Results and Predicted Results:",status = 'primary',
                    DT::DTOutput("sample_predictions"),
                    downloadButton("downloadData", "Download Predictions")
              ))
            )
  )
) #--- end body
#NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN------- ui components
ui <- dashboardPage(skin = "purple",header = header, 
                    sidebar = sidebar,
                    body = body) 
##################################################################################################################
#NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN--------- Create Server -------------------
##################################################################################################################
server <- function(input,output){
  #NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN--- Tabitem 2
  #------------------------------------------------------- Describe data
  output$myplot <- renderPlot({
    colm <- as.numeric(input$features)
    counts <- table(D[,colm])
    barplot(counts,col = "green",main = "Histrogram of Student Dataset",xlab = names(D[colm]))
  })
  output$data <- DT::renderDT({
    datatable(
      D,
      rownames = TRUE,
      options = list(
        fixedColumns = TRUE,
        autoWidth = TRUE,
        ordering = FALSE,
        dom = 'tB',
        buttons = c('copy', 'csv', 'excel', 'pdf')
      ),
      class = "display" #if you want to modify via .css
    )
    })
  output$summary <- renderPrint({
    summary(D)
  })
  output$str <- renderPrint({
    str(D)
  })
  #NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN--- Tabitem 3
  #------------------------------------------ generate table
  Data <- reactive({
    df <- data.frame(Name = input$name, ID = input$ID, SIM1=input$SIM1,SELD3=input$SELD3,ANXI3=input$ANXI3,SELD5=input$SELD5,SIM2=input$SIM2,ANXI1=input$ANXI1,TMP3=input$TMP3,SIM3=input$SIM3,ANXI2=input$ANXI2,SIM4=input$SIM4, SCORE = input$score)
    return(df)
  })
  output$yourdata <- DT::renderDataTable({
    Data()
  })
  #--------------------------------
  observeEvent(input$go, {
    #---------------------------------------------------------------------------
    testset <- Data()
    control <- trainControl(method="cv", number=10)
    metric <- "Accuracy"
    set.seed(7)
    fit.rf <- train(SCORE~., data=train_data, method="knn", metric=metric, trControl=control)
    rf_pred <- predict(fit.rf,testset)
    #-----------------------------------------------------
    output$predvalue <- renderPrint(rf_pred)
    output$pl <- renderText({
      if (rf_pred == 0) {
        paste(input$name, 'you are categorized to be a poor learner that need high intervention')
      } else if (rf_pred == 1) {
        paste(input$name, 'you are categorized to be a average learner that is poosible to have risk, so better need intervention')
      } else if (rf_pred == 2) {
        paste(input$name, 'you are categorized to be a good learner that need low intervention')
      } else{
        paste(input$name, 'you are categorized to be a excellent learner that no need intervention')
      }
    })
  })
  #-NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN--- Tabitem 4
  # Function that imports the data file
  #---------------------------------------------------
  predictions <- reactive({
    inFile <- input$file1
    if (is.null(inFile)){
      return(NULL)
    }else{
      withProgress(message = 'Predictions in progress. Please wait ...', {
        input_data =  readr::read_csv(input$file1$datapath, col_names = TRUE)[,c("SIM1","SELD3","ANXI3","SELD5","SIM2","ANXI1","TMP3","SIM3","ANXI2","SIM4","SCORE")]
        names(input_data)[ncol(input_data)] <- "Actual Classes"
        control <- trainControl(method="cv", number=10)
        metric <- "Accuracy"
        # Random Forest
        
        set.seed(7)
        fit.rf <- train(SCORE~., data=train_data, method="rf", metric=metric, trControl=control)
        prediction = predict(fit.rf, input_data)
        input_data_with_prediction = cbind(input_data,prediction )
        names(input_data_with_prediction)[ncol(input_data_with_prediction)] <- "Predicted Classes"
        input_data_with_prediction
        
      })
    }
    #----------------------------------------------------------
  })
  #Pred_data <- predictions
  # Downloadable csv of selected dataset ----
  output$downloadData <- downloadHandler(
    filename = function() {
      paste(input$predictions() , "predictions.csv", sep = "")
    },
    content = function(file) {
      write.csv(predictions(), file, row.names = FALSE)
    })
#-------------------------------------------------------------------  
  output$sample_predictions = DT::renderDT({     
     predictions()
  })
  output$myplot4 <- renderPlot({
    req(input$file1)
    df <- read.csv(input$file1$datapath)[,c("SIM1","SELD3","ANXI3","SELD5","SIM2","ANXI1","TMP3","SIM3","ANXI2","SIM4","SCORE")]
    colm <- as.numeric(input$features4)
    counts <- table(df[,colm])
    barplot(counts,col = "green",main = "Histrogram of Student Dataset",xlab = names(df[colm]))
  })
 # output$corrplot4 <- DT::renderDT({
#    req(input$file1)
#    df <- read.csv(input$file1$datapath)[,c("SIM1","SELD3","ANXI3","SELD5","SIM2","ANXI1","TMP3","SIM3","ANXI2","SIM4","SCORE")]
#    return(df)
#  })
  output$corrplot4 <- renderPlot({
    req(input$file1)
    df <- read.csv(input$file1$datapath)[,c("SIM1","SELD3","ANXI3","SELD5","SIM2","ANXI1","TMP3","SIM3","ANXI2","SIM4","SCORE")]
    df <- sapply(df,as.numeric)
    M <- cor(df)
    corrplot::corrplot.mixed(M,lower.col = "black", number.cex = .7)
  })
  output$summary4 <- renderPrint({
    req(input$file1)
    df <- read.csv(input$file1$datapath)[,c("SIM1","SELD3","ANXI3","SELD5","SIM2","ANXI1","TMP3","SIM3","ANXI2","SIM4","SCORE")]
    str(df)
  })
  #-------------------------------------------------------------------------------
} #--------------------------- end server
##################################################################################################################
#NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN--------- Create Shiny App ----------------
##################################################################################################################

# Create Shiny app ----
shinyApp(ui = ui, server = server)


