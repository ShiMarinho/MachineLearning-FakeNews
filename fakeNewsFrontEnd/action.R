#install.packages("readtext")
#library("readtext")
x<-"D:/RWebProject/ShinyApp-ver1/ver03/readData.txt"
y<-readtext(x)
print(y$text)
s<-utf8ToInt(y$text)
#barplot(s)
#pie(s,col=rainbow(length(s)))
#hist(s,xlab = "Weight",col = "yellow")
#print(s)
z<-"D:/RWebProject/ShinyApp-ver1/ver03/result.txt"
sink(z)
cat(s)
sink()
