$(document).ready(function(){
  var metadata = $.getJSON("js/portraits.json");
  console.log(metadata);
  var i = 0;

    $(function () {
        $('#YearRange1').on('input change', function () {
            $('#startyear').text("Start Year - " + $(this).val());
            if($('#YearRange1').val() > $('#YearRange2').val()){
                $('#searcherror').text("ERROR: Start Year greather than End Year.");
            }else{
                $('#searcherror').text("");
            }
        });
    });
    $(function () {
        $('#YearRange2').on('input change', function () {
            $('#endyear').text("End Year - " + $(this).val());
            if($('#YearRange1').val() > $('#YearRange2').val()){
                $('#searcherror').text("ERROR: Start Year greather than End Year.");
            }else{
                $('#searcherror').text("");
            }
        });
    });

    $('#m1').on({
        'click': function(){
            window.location = 'search2.html';
            // $('#exampleRadios1').attr("checked", true);
        }
    })
    
    $('#randomz').on({
        'click': function(){
           $('#z1').val(Math.round(Math.random()*10)/10)
           $('#z2').val(Math.round(Math.random()*10)/10)
           $('#z3').val(Math.round(Math.random()*10)/10)
           $('#z4').val(Math.round(Math.random()*10)/10)
           $('#z5').val(Math.round(Math.random()*10)/10)
           $('#z6').val(Math.round(Math.random()*10)/10)
           $('#z7').val(Math.round(Math.random()*10)/10)
           $('#z8').val(Math.round(Math.random()*10)/10)
           $('#z9').val(Math.round(Math.random()*10)/10)
           $('#z10').val(Math.round(Math.random()*10)/10)
           $('#z11').val(Math.round(Math.random()*10)/10)
           $('#z12').val(Math.round(Math.random()*10)/10)
           $('#z13').val(Math.round(Math.random()*10)/10)
           $('#z14').val(Math.round(Math.random()*10)/10)
           $('#z15').val(Math.round(Math.random()*10)/10)
        }
    })

    $('#loadportrait').on({
        'click': function(){
            i += 1
            $('#reconin').attr('src','Images/' + metadata.responseJSON[i]['filename']);
        }
    })

});
