SAY_TEXT ?= hello world
KITTEN_TTS_BIN ?= target/debug/kitten-tts

.PHONY: say
say:
	@test -x "$(KITTEN_TTS_BIN)" || cargo build
	KITTEN_TTS_BIN="$(KITTEN_TTS_BIN)" examples/kitten-say.sh "$(SAY_TEXT)"
