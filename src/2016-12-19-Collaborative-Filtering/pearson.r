norvig 	<- c(2, 6, 3, 8, 8, 6)
hinton 	<- c(6, 4, 4, 2, 7, 8)
navarro <- c(6, 4, 4, 0, 8, 4)
ullman 	<- c(4, 6, 8, 5, 9, 8)
aho 	<- c(5, 8, 8, 8, 10, 9)
carmack <- c(6, 7, 8, 4, 7, 8)

# A function to plot both points and linear regression line
plot_pearson <- function(x, y, xname, yname) {
	xparse <- strsplit(xname, " ")
	xlname <- xparse[[1]][length(xparse[[1]])]

	yparse <- strsplit(yname, " ")
	ylname <- yparse[[1]][length(yparse[[1]])]


	title = paste("Comparing the ratings given by", xlname, "and", ylname)
	plot(x, y, pch=16, cex=1.3, col="blue", main=title, xlab=xname, ylab=yname)
	abline(lm(y ~ x), col="red", lty=2)
}

# The positive correlation example
positive <- function() {
	
	x <- ullman
	y <- carmack

	xname <- "Jeffrey Ullman"
	yname <- "John Carmack"

	plot_pearson(x, y, xname, yname)
	
}

# The negative correlation example
negative <- function() {
	
	x <- navarro
	y <- norvig

	xname <- "Gonzalo Navarro"
	yname <- "Peter Norvig"

	plot_pearson(x, y, xname, yname)
}

# The almost-zero correlation example
zero <- function() {
	
	x <- navarro
	y <- aho

	xname <- "Gonzalo Navarro"
	yname <- "Alfred Aho"

	plot_pearson(x, y, xname, yname)
}