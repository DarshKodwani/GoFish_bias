import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
from email.MIMEBase import MIMEBase
from email import encoders


#Send the email when the script is run                                                               
sender = 'darsh1993@gmail.com'
password = 'Cr7redevilg'
receivers = ['ddkcdk@yahoo.co.uk']

msg = MIMEMultipart()

msg["Subject"] = "Plots of NV isocurvature fluctuations for 50ks"
body = "Lensing GPD"

msg.attach(MIMEText(body, 'plain'))

filename = "in_out.py"
attachment = open("/users/dkodwani/Software/CCL/GoFish_SSC2/in_out.py", "rb")

part = MIMEBase('application', 'octet-stream')
part.set_payload((attachment).read())
encoders.encode_base64(part)
part.add_header('Content-Disposition', "attachment; filename= %s" % filename)

msg.attach(part)

s = smtplib.SMTP('smtp.gmail.com', 587)
s.starttls()
s.login(sender, password)
text=msg.as_string()
s.sendmail(sender, receivers, text)
