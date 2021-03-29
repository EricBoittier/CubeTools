from jinja2 import Template


for a in range(10):
	for b in range(10):
		for c in range(10):
			

			test_agr = open("test.job", 'r').read()
			# Create Template Object
			template = Template(test_agr)
			# Render HTML Template String
			test_agr_template = template.render(a = a, b = b, c = c)
			f = open("Water_{}_{}_{}.slurm".format(a, b, c), "w")
			f.writelines(test_agr_template)

