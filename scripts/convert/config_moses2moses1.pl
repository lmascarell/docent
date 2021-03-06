#!/usr/bin/perl
#
# convert Moses configuration files from version 2 to version 1
# (this is not complete and quite ad-hoc!)
#

use strict;

my %config=();
my %features=();
my %weights=();

my $section = undef;

# read Moses config file version 2

while (<>){
    next unless (/\S/);
    next if (/^\#/);
    chomp;
    if (/^\[(.*)\]$/){
	$section=$1;
    }
    elsif (/^([0-9].*)$/){
	push(@{$config{$section}},$1);
    }
    elsif ($section eq 'feature'){
	my @params = split(/\s+/);
	my $type = shift(@params);
	my %feat=();
	foreach (@params){
	    my ($key,$val) = split(/\=/);
	    $feat{$key} = $val;
	}
	%{$features{$type}{$feat{name}}} = %feat;
    }
    elsif ($section eq 'weight'){
	my ($name,$scores) = split(/\s*\=\s*/);
	@{$weights{$name}} = split(/\s+/,$scores);
    }
}


# write Moses configuration version 1

foreach (keys %config){
    print "[$_]\n";
    print join("\n",@{$config{$_}});
    print "\n\n";
}

$section = undef;
my $useAlgInfo = 0;

foreach my $f (sort keys %features){
    if ($f=~/KENLM/){
	unless ($section eq 'lm'){
	    print "[lmodel-file]\n";
	    $section = 'lm';
	}
	foreach my $n (sort keys %{$features{$f}}){
	    my %l = %{$features{$f}{$n}};
	    print "8 $l{factor} $l{order} $l{path}\n";
	}
	print "\n";
    }
    elsif ($f=~/PhraseDictionaryBinary/){
	unless ($section eq 'pt'){
	    print "[ttable-file]\n";
	    $section = 'pt';
	}
	foreach my $n (sort keys %{$features{$f}}){
	    my %l = %{$features{$f}{$n}};	    
	    print "1 $l{'input-factor'} $l{'output-factor'} ";
	    print $l{'num-features'}+1," $l{path}\n";

	    ## set word alignment flag for the phrase table
	    ## if the .wa files exist 
	    ## (TODO: do we need this for each ttable?)
	    if (-e $l{path}.'.binphr.srctree.wa'){
		$useAlgInfo++;
	    }

	}
	print "\n";
    }
    elsif ($f=~/LexicalReordering/){
	unless ($section eq 'd'){
	    print "[distortion-file]\n";
	    $section = 'd';
	}
	foreach my $n (sort keys %{$features{$f}}){
	    my %l = %{$features{$f}{$n}};	    
	    print "$l{'input-factor'}-$l{'output-factor'} $l{type} ";
	    print $l{'num-features'}," $l{path}\n";
	}
	print "\n";
    }
}


$section = undef;
my $nrPt = 0;
foreach my $f (sort keys %features){
    if ($f=~/KENLM/){
	unless ($section eq 'lm'){
	    print "[weight-l]\n";
	    $section = 'lm';
	}
	foreach my $n (sort keys %{$features{$f}}){
	    print join("\n", @{$weights{$n}});
	}
	print "\n\n";
    }
    elsif ($f=~/PhraseDictionaryBinary/){
	unless ($section eq 'pt'){
	    print "[weight-t]\n";
	    $section = 'pt';
	}
	foreach my $n (sort keys %{$features{$f}}){
	    $nrPt++;
	    print join("\n", @{$weights{$n}});
	    print "\n";
	    print $weights{'PhrasePenalty0'}[0],"\n";
	}
	print "\n";
    }
    elsif ($f=~/LexicalReordering/){
	unless ($section eq 'd'){
	    print "[weight-d]\n";
	    $section = 'd';
	}
	print $weights{'Distortion0'}[0],"\n";
	foreach my $n (sort keys %{$features{$f}}){
	    print join("\n", @{$weights{$n}});
	}
	print "\n\n";
    }
}

if ($useAlgInfo){
    print "[use-alignment-info]\n1\n\n";
}

print "[weight-w]\n";
print $weights{'WordPenalty0'}[0],"\n\n";

print "[distortion-limit]\n6\n\n";

print "[ttable-limit]\n";
foreach (1..$nrPt){
    print "20\n";
}
