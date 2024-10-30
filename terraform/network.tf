// BEGIN: Create a simple vpc
resource "aws_default_vpc" "apg_sage_vpc" {
	tags = {
		Name = "APG Default VPC"
	}
}
// END: Create a simple vpc

// BEGIN: Create a simple subnet
resource "aws_default_subnet" "apg_sage_subnet" {
	availability_zone = "us-east-1c" # substitute for your own region
	tags = {
		Name = "Default subnet for us-east-1c"
	}
}
// END: Create a simple subnet

// BEGIN: Create a simple security group
resource "aws_default_security_group" "apg_sage_security_group" {
//name_prefix = "apg-sage-security-group"
vpc_id = aws_default_vpc.apg_sage_vpc.id

	ingress {
		protocol = -1
		self = true
		from_port = 0
		to_port = 0
	}
	
	egress {
		from_port = 0
		to_port = 0
		protocol = "-1"
		cidr_blocks = [aws_default_subnet.apg_sage_subnet.cidr_block]
	}
}
// END: Create a simple security group
