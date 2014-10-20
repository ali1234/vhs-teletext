HOW TO USE
----------

If you want to try it I suggest starting with Ubuntu in a virtual machine. You'll need to install some packages:

	sudo apt-get install python-numpy python-scipy tv-fonts xterm git

Then run an xterm like this:

	/usr/bin/xterm -fg white -bg black -fn teletext &

Then checkout source and run on the example data like this:

	git clone git://github.com/ali1234/vhs-teletext.git

All the things you need should be available to run this on Windows, but I have not tried it.



In detail:

Step 1 - Capture teletext packets

(If you are running in a VM you probably won't be able to use capture hardware.
In that case you must capture in the host OS and then copy in the vbi files.)

Compile dumpvbi:

	sudo apt-get install build-essential
	gcc -o dumpvbi dumpvbi.c

Create a folder for captures:

	mkdir cap0001
	cd cap0001
        mkdir vbi
        cd vbi

Do capture:

	../../dumpvbi

You will need at least 15-30 minutes for good results. 
End capture with ctrl-c and then go back to previous directory:

	cd ../..

This produces raw binary files starting at 00000000.vbi. Each file contains 32 
vbi lines sampled from a single frame. Each line is 2048 samples.




Step 2 - Run recovery

Run vbi.py on the sampled data:

	./vbi.py cap0001/

Step 3 - Page splitter

	./t42cat.py cap0001/t42/ 2> /dev/null | ./pagesplit.py cap0001/pages/


Step 4 - Squash subpages to html:

	./subpagesquash.py cap0001/pages/ cap0001/html-pages/

Outputs hyperlinked html files for each page. You can run this while the page
splitter is still working, and you can run it over and over to get the newest
results.




Step 5 - Copy font and CSS:

	cp teletext2.ttf teletext4.tff teletext.css cap0001/html-pages/

This is needed to make the html files render properly. You can now open 100.html
in firefox from the file manager, or open it from the terminal with:

	xdg-open cap0001/html-pages/100.html


HOW IT WORKS
------------

Teletext is encoded as a non-return-to-zero signal with two levels representing
one and zero. This is a fancy way of saying that a line of teletext data is
a sequence of black and white "pixels" in the TV signal. Of course, since the
signal is analogue there are no individual pixels, the signal is continuous.
But you can imagine that there are pixels in the idealized "perfect" signal.

The problem of decoding teletext from a VHS recording is that VHS bandwidth is
lower than teletext bandwidth. This means that the signal is effectively low
pass filtered, which in terms of an image is equivalent to gaussian blurring.

There are methods for reversing gaussian blur, but they are designed to work
with general image data. In the case of teletext we only have black or white
levels, so these methods are not optimal. We can exploit the limitations on
the input in order to get a better result. We can also exploit information
about the protocol to further improve efficiency and accuracy.

When the black and white signal is blurred, the individual pixels are blurred 
in to each other. This makes the signal unreadable using normal methods, because
instead of a clean sequence like "1010" you something close to "0.5 0.5 0.5 0.5".
But all is not lost, because a sequence like "1111" or "0000" will be the same
after blurring. So if you see a signal like "0.5 0.7 1.0 1.0" you can guess that
the original was probably "0 1 1 1" or "0 0 1 1".  

What this software does is take the sampled blurred signal and tries to guess
what original data would produce the sampled result. It does this by applying
a guassian filter approximating the VHS encoding to the guess, and then testing
the result against the observed data.

There are 45 bytes in each teletext line, so the space of possible guesses is
2^(45*8) which is a very big number, which makes trying every guess completely
impractical. However there are ways to reduce this number:

FOUR RULES
----------

1. Nearly all bytes have a parity bit which means there are only 128 possible
combinations instead of 256.

2. Some bytes are hamming encoded. These have even fewer possible combinations.

3. The first three bytes in the signal are always the same. We can use this
to find the start of the signal in the sample data (it moves a bit in each
line, but the width is always the same.)

4. The protocol itself defines rules about which bytes are allowed in which
positions, reducing the problem space further.


GUESSING ALGORITHM
------------------

1. We start with a completely empty guess, all bits are unknown.

2. We fill in the first three known bytes, and apply the gaussian filter,
and then correlate the result with the samples to find the beginning of
the signal.

3. Take the samples input and find areas which are close to 0 or 1. These
represent a run of 0s or 1s in the original data, so we fix these values
in the guess. This typically fills in about 5-10 of the original bits.

4. Next we have to guess what all the other values are. This is done recursivly,
one byte at a time, from the beginning of the signal to the end. Like so:

a. For the current byte position, apply the FOUR RULES to restrict the number
of guesses. From the resulting set of possible bytes, remove any that contradict
bits we filled in during step 3.

b. Try each possible byte guess in sequence and find the one that most closely
matches the samples.

c. Fill the best result in to the guess and go to the next byte.

4. During the first pass, the subsequent bytes after the one we are currently
guessing are completely wrong. So we go back to the start of the signal and
try every possible byte again, but this time we have a better approximation in
the subsequent bytes.

We keep repeating this until the guess converges - ie it doesn't change
at all during a pass. This usually takes about 5 - 10 passes.

Once the guess has converged, this is the decoded teletext bit data. This is
written out to a file for more processing later.

It should be noted that due to only trying possible bytes in the guess, the first
stage always outputs valid teletext packets with correct parity etc, so we can't
do further error detection, since no parity or hamming errors can ever be produced.


LINE CLEANUP
------------

The guessing algorithm output lots of teletext packets, but they will still not be    
perfect (even though they are valid, they aren't necessarily correct.)

Since the teletext pages are broadcast on a loop, any recording of more than a
few minutes will have multiple copies of every packet. A hamming distance check
it used to determine if any two output guesses are probably the same line.

This means, if two packets are received that only differ at a couple of bytes,
they are assumed to be the same.

All versions of the same packet are compared, and for each byte, the most frequent
decoding is used so for example if you had these inputs:

HELLO
HELLP
MELLO

Then the result "HELLO" would be decoded, since those are the most frequent bytes
in this position. For this to work well, you need a lot of copies of every packet.
This procedure only considers the visible data, not row numbers, because the same
packet may appear on many pages if it is a repeated header graphic for example.

Also whitespace is ignored, because many packets are all whitespace except for eg
a subpage number, and these should not be combined. In order to match, a packet
must have more than X non-whitespace characters the same, and less the Y different.

After applying this check, each packet in the input is replaced by it's "cleaned"
version in the output, because sequence of packets is important.


FINDING HEADERS
---------------

The guessing algorithm output lots of teletext lines, but they will still not be
perfect. We need to rebuild it into pages.

Each teletext page starts off with a special header called packet zero with a
special format. This format is considered as part of the FOUR RULES above, so
such headers are more likely to be decoded correctly. The headers usually have a
very specific format containing the broadcaster ident and a clock, which means
that the "possible bytes" can be highly refined eg "last byte is always a digit".

A teletext page is defined as all the packets which follow the header, until the
next header. This is why packet sequence is preserved at the previous step.

The rules for headers are defined in finders.py and if you want to decode for
a new broadcaster you need to define a finder for it. Do this by running the
previous stages first, then examining the output. The headers will not be detected
but they will still be output like normal packets and you should be able to
reconstruct a pristine one by hand (approx 4% of all lines are headers so you
should find plenty of examples.)

After you program in the rules for the header packet, the software will find
them much more easily and with higher accuracy.


PAGE REBUILD
------------

The data is considered in sequence and for each header, subsequent packets with
the same magazine number are considered part of the page.

Once anew header is received, all the previously received packets are passed through
a sanity check. If most of the lines are present (1-26 or so) then the page is
considered complete and written out to disk. If many lines are missing then the page is
discarded as unrecoverable.

If you have a lot of input data (and you should) then you will get multiple examples
of each page. They are all combined again using hamming distance like was done with
individual packets, with the result being considered the best version of the page.

There are also rules which recognize graphical elements in pages such as logos. For
better results you can teach new rules about these, which will improve decodes.
They are in fragment.py and they work kind of like finders, but in 2 dimensions.


FINAL NOTES
-----------

For best results the software needs to be taught lots of things about the
data you are decoding:

* Broadcaster header format (finders.py)
* Common logos (fragment.py)

It also needs a large amount of input data due to discarding lots of bad, 
unrecoverable parts. You need up to half an hour of real-time sampled data in 
order to get best results. Unfortunately the decoder doesn't run in anything
like real-time, and this might take a month to do the first stage guess filtering
on a quad core PC.

The software expects the sampled lines to be 2048 8-bit samples. This is what
bt8x8 produces. Other capture hardware produces 1440 samples. In this case,
you need to resample to 2048 samples when loading the data.
