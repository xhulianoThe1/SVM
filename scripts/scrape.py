#remember to install twint through git
#must manually change handles, and manually stop run
import twint
import csv
c = twint.Config()
c.Username = "markets"
c.Store_csv = True
c.Output = "Bloomberg.csv"

twint.run.Search(c)
