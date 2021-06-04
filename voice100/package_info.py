# Copyright (C) 2021 Katsuya Iida. All rights reserved.

MAJOR = 0
MINOR = 0
PATCH = 1
PRE_RELEASE = 'a'

# Use the following formatting: (major, minor, patch, pre-release)
VERSION = (MAJOR, MINOR, PATCH, PRE_RELEASE)

__shortversion__ = '.'.join(map(str, VERSION[:3]))
__version__ = '.'.join(map(str, VERSION[:3])) + ''.join(VERSION[3:])

__package_name__ = 'voice100'
__contact_names__ = 'Katsuya Iida'
__contact_emails__ = 'katsuya.iida@gmail.com'
__homepage__ = 'https://github.com/kaiidams/voice100/'
__repository_url__ = 'https://github.com/kaiidams/voice100/'
__download_url__ = 'https://github.com/kaiidams/voice100/releases'
__description__ = 'Voice100 - Tiny TTS'
__license__ = 'MIT'
__keywords__ = 'deep learning, machine learning, pytorch, torch, asr, tts, speech'
