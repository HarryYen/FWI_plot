#!/bin/bash

ps -ef | grep harry | grep wav_plot | grep -v grep | awk '{print $2}'| xargs kill -9
