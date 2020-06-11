import QtQuick 2.12
import QtGraphicalEffects 1.12


Rectangle {
    property var borderSize: 19 * ttzoom
    width: teletext.width + borderSize * 4
    height: teletext.height + borderSize * 2
    border.width: borderSize
    border.color: "black"
    color: "black"
    GridView {
        id: teletext
        width: (320 * ttzoom)
        height: 250 * ttzoom
        cellHeight: 10 * ttzoom
        cellWidth: 8 * ttzoom
        x: parent.borderSize * 2
        y: parent.borderSize
        interactive: false
        clip: true
        model: ttmodel

        delegate: Rectangle {
            color: ttpalette[display.bg]
            Text {
                renderType: Text.NativeRendering
                color: ttpalette[display.fg]
                text: display.text
                font: ttfonts[display.width-1][display.height-1]
                MouseArea {
                    anchors.fill: parent
                    onClicked: teletext.currentIndex = index
                }
                SequentialAnimation on opacity {
                    loops: -1
                    running: display.flash
                    alwaysRunToEnd: true
                    PropertyAction { value: 0 }
                    PauseAnimation { duration: 333 }
                    PropertyAction { value: 1 }
                    PauseAnimation { duration: 1000 }
                }
            }
            height: display.height * 10 * ttzoom
            width: display.width * 8 * ttzoom
            clip: true
            visible: display.visible
        }

        highlightMoveDuration: 0
        highlight: ShaderEffect {
            z: 1
            property variant source: ShaderEffectSource { sourceItem: teletext.currentItem }
            fragmentShader: "
                uniform lowp sampler2D source;
                uniform lowp float qt_Opacity;
                varying highp vec2 qt_TexCoord0;
                varying lowp vec3 qt_FragCoord0;
                void main(void)
                {
                    lowp vec4 tex = texture2D(source, qt_TexCoord0);
                    gl_FragColor = vec4(1-tex.r, 1-tex.g, 1-tex.b, 1);
                }
            "
            SequentialAnimation on opacity {
                loops: -1
                running: true
                alwaysRunToEnd: true
                PropertyAction { value: 1 }
                PauseAnimation { duration: 166 }
                PropertyAction { value: 0 }
                PauseAnimation { duration: 166 }
            }
        }

        layer.enabled: tteffect && (ttfonts[0][0].pixelSize > 10)
        layer.effect: ShaderEffect {
            fragmentShader: "
                    uniform lowp sampler2D source;
                    uniform lowp float qt_Opacity;
                    varying highp vec2 qt_TexCoord0;
                    varying lowp vec3 qt_FragCoord0;
                    void main() {
                        lowp vec4 tex = texture2D(source, qt_TexCoord0);
                        int ttzoom = " + ttzoom + ";
                        int row = int(gl_FragCoord.y) % ttzoom;
                        gl_FragColor = (0 < row && (row < 2 || row < (ttzoom-1))) ? tex : tex*0.6;
                    }
                "
        }
    }
    layer.enabled: tteffect && (ttzoom > 1)
    layer.effect: GaussianBlur {
        radius: 0.75*ttzoom
    }
}


