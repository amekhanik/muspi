from muspi import *
REVERB = False


#initialize all notes
nc = Note('C', 4)
nc_sh = Note('C#', 4)
nd = Note('D', 4)
nd_sh = Note('D#', 4)
ne = Note('E', 4)
nf = Note('F', 4)
nf_sh = Note('F#', 4)
ng = Note('G', 4)
ng_sh = Note('G#', 4)
na = Note('A', 4)
na_sh = Note('A#', 4)
nb = Note('B', 4)

#initialize notes 1 oct higher
nd5 = nd+12
ne5 = ne+12
nf5 = nf+12
nc5 = nc+12


# Chords for Dogs
Dm9 = Chord(nd,'min9')
Bbadd11 = Chord(na_sh+(-12), 'add11')
A9sus4 = Chord(na_sh+(-12), '9sus4')
Bb7 = Chord(na_sh+(-12), '7')
s = Chord(nc, 'silence')

scale_types = ['maj', 'min','harm min', 'mel min', 'half dim']
scale2play = 2









'''
if REVERB:
#initialize reverb
    wf = wave.open('impulse_cathedral_audacity_trim.wav', 'rb')

    reverb = []
    input_string = wf.readframes(1)
    while input_string != '':
        # Convert string to number
        input_tuple = unpack('i', input_string)  # One-element tuple
        reverb.append(input_tuple[0])  # Number
        # Get next frame
        input_string = wf.readframes(1)

    l = Dm9.generate_chord(0.5)
    reverb_chord = fconv(l,reverb)
    output_string = pack('i'*len(reverb_chord), *reverb_chord)
    stream.write(output_string)
    wf.close()
if False:
    Dm9.play_chord(0.5, ['distortion'])
    Bbadd11.play_chord(0.5, ['distortion'])
    A9sus4.play_chord(0.5 , ['distortion'])
    Bb7.play_chord(0.5, ['distortion'])

    Dm9.play_chord(0.5, ['flanger'])
    Bbadd11.play_chord(0.5, ['flanger'])
    A9sus4.play_chord(0.5 , ['flanger'])
    Bb7.play_chord(0.5, ['flanger'])

    Dm9.play_chord(0.5, ['flanger','staccato'])
    Bbadd11.play_chord(0.5, ['flanger','staccato'])
    A9sus4.play_chord(0.5, ['flanger','staccato'])
    Bb7.play_chord(0.5, ['flanger','staccato'])

while False:
    # bar 1
    Dm9.play_chord(.125)
    Dm9.play_chord(.125 / 2)
    Dm9.play_chord(.125 / 2)
    s.play_chord(.125 / 2)
    s.play_chord(.125 / 2)

    Dm9.play_chord(.125)
    Dm9.play_chord(.125 / 2)
    Dm9.play_chord(.125 / 2)
    s.play_chord(.125 / 2)
    s.play_chord(.125 / 2)

    Dm9.play_chord(.125/2)
    Dm9.play_chord(.125/2)
    s.play_chord(.125/2)
    s.play_chord(.125/2)

    # bars 2-4
    for i in range(3):
        Dm9.play_chord(.125*1.5)
        Dm9.play_chord(.125/2)

        s.play_chord(.125 / 2)
        s.play_chord(.125/2)
        Dm9.play_chord(.125*1.5)

        Dm9.play_chord(.125/2)
        s.play_chord(.125/2)
        s.play_chord(.125/2)

        Dm9.play_chord(.125/2)
        Dm9.play_chord(.125/2)
        s.play_chord(.125/2)
        s.play_chord(.125/2)
    # bar 5-8
    for i2 in range(4):
        Bbadd11.play_chord(.125*1.5)
        Bbadd11.play_chord(.125/2)

        s.play_chord(.125 / 2)
        s.play_chord(.125/2)
        Bbadd11.play_chord(.125*1.5)

        Bbadd11.play_chord(.125/2)
        s.play_chord(.125/2)
        s.play_chord(.125/2)

        Bbadd11.play_chord(.125/2)
        Bbadd11.play_chord(.125/2)
        s.play_chord(.125/2)
        s.play_chord(.125/2)

    # bar 9-12
    for i3 in range(4):
        A9sus4.play_chord(.125 * 1.5)
        A9sus4.play_chord(.125 / 2)

        s.play_chord(.125 / 2)
        s.play_chord(.125 / 2)
        A9sus4.play_chord(.125 * 1.5)

        A9sus4.play_chord(.125 / 2)
        s.play_chord(.125 / 2)
        s.play_chord(.125 / 2)

        A9sus4.play_chord(.125 / 2)
        A9sus4.play_chord(.125 / 2)
        s.play_chord(.125 / 2)
        s.play_chord(.125 / 2)

    # bars 13-16
    for i4 in range(4):
        Bb7.play_chord(.125 * 1.5)
        Bb7.play_chord(.125 / 2)

        s.play_chord(.125 / 2)
        s.play_chord(.125 / 2)
        Bb7.play_chord(.125 * 1.5)

        Bb7.play_chord(.125 / 2)
        s.play_chord(.125 / 2)
        s.play_chord(.125 / 2)

        Bb7.play_chord(.125 / 2)
        Bb7.play_chord(.125 / 2)
        s.play_chord(.125 / 2)
        s.play_chord(.125 / 2)

'''
# Creates notes and chords to be played for Song of storms
a =  Chord(na, 'note')
d = Chord(nd, 'note')
e = Chord(ne, 'note')
g = Chord(ng, 'note')
f = Chord(nf, 'note')
c5 = Chord(nc5, 'note')
d5 = Chord(nd5, 'note')
e5 = Chord(ne5, 'note')
f5 = Chord(nf5, 'note')

dmin = Chord(nd, 'min')
emin = Chord(ne, 'min')
fmaj = Chord(nf, 'maj')
amaj = Chord(na, 'maj')
Bb_maj7 = Chord(na_sh, 'maj7')

# intro chords
nd3 = nd + (-12)
ne3 = ne + (-12)
nf3 = nf + (-12)
nb3 = nb + (-12)

d3 = Chord(nd3, 'note')
e3 = Chord(ne3, 'note')
f3 = Chord(nf3, 'note')


d3min  =  Chord(nd3, 'min')
e3min  =  Chord(ne3, 'min')
f3maj  =  Chord(nf3, 'maj')

# Song of storms
# intro
# bar 1
bar_l = []
bar_l.append([d3,.25,['distortion']])
bar_l.append([d3min,.25,['distortion']])
bar_l.append([d3min,.25,['distortion']])

bar_r = []
bar_r.append([dmin, .75, ['distortion']])

play_bar(bar_l, bar_r)

# bar 2
bar_l = []
bar_l.append([d3,.25,['distortion']])
bar_l.append([e3min,.5,['distortion']])

bar_r = []
bar_r.append([emin, .75, ['distortion']])

play_bar(bar_l, bar_r)

# bar 3
bar_l = []
bar_l.append([f3,.25,['distortion']])
bar_l.append([f3maj,.25,['distortion']])
bar_l.append([f3maj,.25,['distortion']])

bar_r = []
bar_r.append([fmaj, .75, ['distortion']])

play_bar(bar_l, bar_r)

# bar 4
bar_l = []
bar_l.append([d3,.25,['distortion']])
bar_l.append([e3min,.5,['distortion']])

bar_r = []
bar_r.append([emin, .75, ['distortion']])

play_bar(bar_l, bar_r)

# bar 5
bar_l = []
bar_l.append([d3,.25,['distortion']])
bar_l.append([d3min,.25,['distortion']])
bar_l.append([d3min,.25,['distortion']])

bar_r = []
bar_r.append([dmin, .75, ['distortion']])

play_bar(bar_l, bar_r)

# bar 6
bar_l = []
bar_l.append([d3,.25,['distortion']])
bar_l.append([e3min,.5,['distortion']])

bar_r = []
bar_r.append([emin, .75, ['distortion']])

play_bar(bar_l, bar_r)

# bar 7
bar_l = []
bar_l.append([f3,.25,['distortion']])
bar_l.append([f3maj,.25,['distortion']])
bar_l.append([f3maj,.25,['distortion']])

bar_r = []
bar_r.append([fmaj, .75, ['distortion']])

play_bar(bar_l, bar_r)

# bar 8
bar_l = []
bar_l.append([d3,.25,['distortion']])
bar_l.append([e3min,.5,['distortion']])

bar_r = []
bar_r.append([emin, .75, ['distortion']])

play_bar(bar_l, bar_r)


# bar 1
bar_l = []
bar_l.append([d,.125,['distortion']])
bar_l.append([f,.125,['distortion']])
bar_l.append([d5,.5,['distortion']])

bar_r = []
bar_r.append([dmin, .75, ['distortion']])

play_bar(bar_l, bar_r)

# bar 2
bar_l = []
bar_l.append([d,.125,['distortion']])
bar_l.append([f,.125,['distortion']])
bar_l.append([d5,.5,['distortion']])

bar_r = []
bar_r.append([emin, .75, ['distortion']])

play_bar(bar_l, bar_r)

# bar 3
bar_l = []
bar_l.append([e5, .25*1.5, ['distortion']])
bar_l.append([f5, .125, ['distortion']])
bar_l.append([e5, .125, ['distortion']])
bar_l.append([f5, .125, ['distortion']])

bar_r = []
bar_r.append([fmaj, .75, ['distortion']])

play_bar(bar_l, bar_r)

# bar 4
bar_l = []
bar_l.append([e5, .125, ['distortion']])
bar_l.append([c5,  .125, ['distortion']])
bar_l.append([a,.5, ['distortion']])

bar_r = []
bar_r.append([emin, .75, ['distortion']])

play_bar(bar_l, bar_r)

# bar 5
bar_l = []
bar_l.append([a, .25, ['distortion']])
bar_l.append([d,  .25, ['distortion']])
bar_l.append([f,.125, ['distortion']])
bar_l.append([g,.125, ['distortion']])

bar_r = []
bar_r.append([Bb_maj7, .75, ['distortion']])

play_bar(bar_l, bar_r)

# bar 6
bar_l = []
bar_l.append([a, .75, ['vibrato', 'bend_half','distortion']])

bar_r = []
bar_r.append([fmaj, .75, ['distortion']])
play_bar(bar_l, bar_r)


# bar 7
bar_l = []
bar_l.append([a, .25, ['distortion']])
bar_l.append([d,  .25, ['distortion']])
bar_l.append([f,.125, ['distortion']])
bar_l.append([g,.125, ['distortion']])

bar_r = []
bar_r.append([Bb_maj7, .75, ['distortion']])

play_bar(bar_l, bar_r)

# bar 8
bar_l = []
bar_l.append([e, .75, ['vibrato', 'bend_half','distortion']])

bar_r = []
bar_r.append([amaj, .75, ['distortion']])
play_bar(bar_l, bar_r)

# bar 9
bar_l = []
bar_l.append([d,.125,['distortion']])
bar_l.append([f,.125,['distortion']])
bar_l.append([d5,.5,['distortion']])

bar_r = []
bar_r.append([dmin, .75, ['distortion']])

play_bar(bar_l, bar_r)

# bar 10
bar_l = []
bar_l.append([d,.125,['distortion']])
bar_l.append([f,.125,['distortion']])
bar_l.append([d5,.5,['distortion']])

bar_r = []
bar_r.append([emin, .75, ['distortion']])

play_bar(bar_l, bar_r)

# bar 11
bar_l = []
bar_l.append([e5, .25*1.5, ['distortion']])
bar_l.append([f5, .125, ['distortion']])
bar_l.append([e5, .125, ['distortion']])
bar_l.append([f5, .125, ['distortion']])

bar_r = []
bar_r.append([fmaj, .75, ['distortion']])

play_bar(bar_l, bar_r)

# bar 12
bar_l = []
bar_l.append([e5, .125, ['distortion']])
bar_l.append([c5,  .125, ['distortion']])
bar_l.append([a,.5, ['distortion']])

bar_r = []
bar_r.append([emin, .75, ['distortion']])

play_bar(bar_l, bar_r)

# bar 13
bar_l = []
bar_l.append([a, .25, ['distortion']])
bar_l.append([d,  .25, ['distortion']])
bar_l.append([f,.125, ['distortion']])
bar_l.append([g,.125, ['distortion']])

bar_r = []
bar_r.append([Bb_maj7, .75, ['distortion']])

play_bar(bar_l, bar_r)

# bar 14
bar_l = []
bar_l.append([a, .5, ['distortion']])
bar_l.append([a, .25, ['distortion']])

bar_r = []
bar_r.append([fmaj, .75, ['distortion']])
play_bar(bar_l, bar_r)

# bars 15&16
bar_l = []
bar_l.append([d, 1.5, ['vibrato', 'bend_full','distortion']])

bar_r = []
bar_r.append([dmin, 1.5, ['distortion']])

play_bar(bar_l, bar_r)



stream.stop_stream()
stream.close()
p.terminate()


